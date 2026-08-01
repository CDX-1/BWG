"""Turn the 50 Hz sensor buffer into windowed features Gemma can reason over.

The single biggest reason the previous prompt was useless: it asked Gemma to
"predict the NEXT most likely failure based on sensor trends" and then sent
29 scalars and a clock stuck at zero. This module produces the trend features
the model was pretending to have.

Windows are 5, 30, and 60 seconds — a spread that lets the same call see
short-lived transients, medium-term drift, and full-buffer context in one
compact payload. Cheap in tokens; falsifiable in outcome.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from parallax.hardware import SensorSample


WINDOW_SECONDS = (5.0, 30.0, 60.0)


@dataclass
class Feature:
    signal: str
    window_s: float
    slope: float          # units per second
    minimum: float
    maximum: float
    variance: float
    threshold_crossings: int
    latest: float


def _window(samples: list[SensorSample], seconds: float) -> list[SensorSample]:
    if not samples:
        return []
    cutoff = samples[-1].received_at - seconds
    return [s for s in samples if s.received_at >= cutoff]


def _slope(values: list[float], times: list[float]) -> float:
    """Linear regression slope; robust to duplicate timestamps."""
    n = len(values)
    if n < 2:
        return 0.0
    t0 = times[0]
    xs = [t - t0 for t in times]
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def _variance(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return sum((v - m) ** 2 for v in values) / (n - 1)


def _crossings(values: list[float], threshold: float) -> int:
    """Number of times the signal crosses `threshold` in either direction."""
    count = 0
    prev = None
    for v in values:
        if prev is not None and ((prev < threshold <= v) or (prev > threshold >= v)):
            count += 1
        prev = v
    return count


def _summarise_signal(
    samples: list[SensorSample],
    attr: str,
    threshold: float,
    getter=None,
) -> list[Feature]:
    features: list[Feature] = []
    getter = getter or (lambda s: getattr(s, attr))
    for w in WINDOW_SECONDS:
        window = _window(samples, w)
        if not window:
            continue
        values = [getter(s) for s in window]
        times = [s.received_at for s in window]
        features.append(Feature(
            signal=attr,
            window_s=w,
            slope=_slope(values, times),
            minimum=min(values),
            maximum=max(values),
            variance=_variance(values),
            threshold_crossings=_crossings(values, threshold),
            latest=values[-1],
        ))
    return features


def cross_correlation(a: list[float], b: list[float]) -> float:
    """Pearson correlation between two aligned signals of the same window.

    Returns 0.0 when either signal is constant — a common case at rest.
    """
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[-n:], b[-n:]
    am, bm = sum(a) / n, sum(b) / n
    num = sum((x - am) * (y - bm) for x, y in zip(a, b))
    da = math.sqrt(sum((x - am) ** 2 for x in a))
    db = math.sqrt(sum((y - bm) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def featurize_hardware_buffer(samples: list[SensorSample]) -> dict:
    """Full feature payload for the tier prompts.

    Compact and JSON-serialisable. Every entry has units in the field name so
    Gemma cannot misread e.g. rate for angle.
    """
    if not samples:
        return {"empty": True, "reason": "no samples in buffer"}

    def gyro_rate(s): return s.gyro_magnitude
    def accel_g(s): return s.accel_magnitude
    def tilt(s): return s.tilt_deg

    features: list[Feature] = []
    features += _summarise_signal(samples, "gyro_rate_deg_s", threshold=15.0, getter=gyro_rate)
    features += _summarise_signal(samples, "accel_g", threshold=1.35, getter=accel_g)
    features += _summarise_signal(samples, "tilt_deg", threshold=25.0, getter=tilt)

    if any(s.temp_c is not None for s in samples):
        temp_samples = [s for s in samples if s.temp_c is not None]
        features += _summarise_signal(temp_samples, "external_temp_c",
                                       threshold=35.0, getter=lambda s: s.temp_c)

    # Cross-signal correlation over the last 30 s of gyro rate vs accel deviation
    window = _window(samples, 30.0)
    gyro_series = [s.gyro_magnitude for s in window]
    accel_dev = [abs(s.accel_magnitude - 1.0) for s in window]
    xcorr = cross_correlation(gyro_series, accel_dev)

    # A running age estimate — the previous prompt sent mission_time=0.0, which
    # made trend claims literally impossible to check.
    duration_s = samples[-1].received_at - samples[0].received_at

    return {
        "sample_count": len(samples),
        "duration_s": round(duration_s, 2),
        "sample_rate_hz": round((len(samples) - 1) / duration_s, 2) if duration_s > 0 else 0.0,
        "cross_correlation_gyro_vs_accel_30s": round(xcorr, 3),
        "features": [
            {
                "signal": f.signal,
                "window_s": f.window_s,
                "slope_per_s": round(f.slope, 4),
                "min": round(f.minimum, 3),
                "max": round(f.maximum, 3),
                "variance": round(f.variance, 4),
                "threshold_crossings": f.threshold_crossings,
                "latest": round(f.latest, 3),
            }
            for f in features
        ],
    }


def featurize_spacecraft_state(
    state,
    samples: list[SensorSample] | None = None,
    sensor_health: dict | None = None,
) -> dict:
    """Produce a compact featurised snapshot to send to the tier prompts.

    Combines the spacecraft simulator state (which is a scalar snapshot) with
    the windowed hardware features when a live board is connected. When the
    board is absent, `hardware_features` is omitted rather than faked. When
    a sensor-health payload is supplied, it is included so the tier prompts
    can reason about loss-of-signal events without a separate call path.
    """
    payload = {
        "mission_time_s": round(getattr(state, "mission_time_s", 0.0), 1),
        "subsystem_health": dict(state.subsystem_health),
        "active_faults": list(state.active_faults),
        # A handful of scalars that don't come from the hardware buffer.
        "scalars": {
            "solar_output_w": round(state.solar_output_w, 1),
            "battery_soc_pct": round(state.battery_soc_pct, 2),
            "bus_voltage_v": round(state.bus_voltage_v, 2),
            "power_draw_w": round(state.power_draw_w, 1),
            "pcu_temp_c": round(state.pcu_temp_c, 1),
            "bus_temp_c": round(state.bus_temp_c, 1),
            "instrument_temp_c": round(state.instrument_temp_c, 1),
            "attitude_error_arcsec": round(state.attitude_error_arcsec, 2),
            "rw_speed_rpm": round(state.rw_speed_rpm, 0),
            "sun_pointing_error_deg": round(state.sun_pointing_error_deg, 3),
            "signal_strength_dbm": round(state.signal_strength_dbm, 2),
            "link_margin_db": round(state.link_margin_db, 2),
            "data_rate_mbps": round(state.data_rate_mbps, 3),
            "adcs_mode": state.adcs_mode,
            "antenna_mode": state.antenna_mode,
            "spectrometer_status": state.spectrometer_status,
        },
    }

    if samples:
        payload["hardware_features"] = featurize_hardware_buffer(samples)

    if sensor_health:
        payload["sensor_health"] = sensor_health

    return payload
