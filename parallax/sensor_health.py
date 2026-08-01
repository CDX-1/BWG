"""Loss-of-signal detection for the ESP32 sensor payload.

The firmware already flags one class of LOS honestly: the ultrasonic ranger
reports −1.0 on timeout. The IMU has no in-band health flag — an I2C freeze
leaves the reader returning the last register values or all-zeros forever,
which reads as "sensor is perfectly stable" downstream. This module is the
Python-side authority that decides when a sensor has actually gone quiet.

Four classes of failure are watched, each with a specific signature:

  link_dead          — no serial frames in >3 s (whole board disconnected)
  imu_stale          — gyro triple identical for >3 s while frames arrive
  imu_flatline       — exact 0.00 on all three axes for many consecutive
                       frames (I2C returning zeros)
  range_persistent   — sustained −1.0 from the ultrasonic ranger for >5 s

A single-shot health evaluation is stateless — it looks at the current
buffer and decides. This is deliberately cheap so we can poll every rerun
without adding cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ── Thresholds ──────────────────────────────────────────────────────────────

LINK_MAX_AGE_S = 3.0        # link considered dead after this many seconds of silence
IMU_STALE_SECONDS = 3.0     # identical gyro triples for this long → frozen
IMU_FLATLINE_COUNT = 60     # consecutive exact-zero frames → I2C bricked
IMU_FLATLINE_EPS = 1e-6
RANGE_NO_ECHO_SECONDS = 5.0 # sustained -1.0 from HC-SR04 → transducer dead
LOW_RATE_HZ = 15.0          # sample rate below this = degraded firmware/link
IMU_MAX_GYRO_DPS = 2500.0   # hardware max is 2000 deg/s at ±2000 setting
IMU_MAX_ACCEL_G = 15.0      # hardware max is 8 g at ±8g setting


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class SensorState:
    name: str
    status: str            # "nominal" | "degraded" | "failed" | "unknown"
    reason: str = ""
    last_seen_s_ago: Optional[float] = None
    fault_id: Optional[str] = None    # populated when this state triggers a fault


@dataclass
class SensorHealthReport:
    link_alive: bool
    sample_rate_hz: float
    per_sensor: dict[str, SensorState]
    active_loss_events: list[str] = field(default_factory=list)

    @property
    def any_failed(self) -> bool:
        return any(s.status == "failed" for s in self.per_sensor.values())

    @property
    def any_degraded(self) -> bool:
        return any(s.status in ("failed", "degraded") for s in self.per_sensor.values())

    def failed_names(self) -> list[str]:
        return [s.name for s in self.per_sensor.values() if s.status == "failed"]

    def summary(self) -> str:
        if not self.link_alive:
            return "LINK DEAD — no frames from the sensor board"
        if self.any_failed:
            names = ", ".join(self.failed_names())
            return f"SENSOR LOSS — {names} failed"
        if self.any_degraded:
            return "SENSORS DEGRADED — see per-sensor detail"
        return "SENSORS NOMINAL"


def _s_ago(t_epoch: float, now: float) -> float:
    return max(0.0, now - t_epoch)


def _nominal(name: str, last_seen_s_ago: float = 0.0) -> SensorState:
    return SensorState(name=name, status="nominal", reason="",
                       last_seen_s_ago=last_seen_s_ago)


# ── Public evaluators ───────────────────────────────────────────────────────

def evaluate_health(link, samples) -> SensorHealthReport:
    """Return a full sensor health report from a SerialLink + its buffer.

    Passing samples in separately lets tests fake the buffer, and lets the
    caller pass a pre-computed history without re-locking the buffer.
    """
    now = time.time()

    if not samples or not link.is_running():
        return SensorHealthReport(
            link_alive=False,
            sample_rate_hz=0.0,
            per_sensor={
                "link": SensorState("link", "failed",
                                     "serial port not connected",
                                     fault_id="sensor_loss_link"),
            },
            active_loss_events=["sensor_loss_link"] if link.is_running() else [],
        )

    latest = samples[-1]
    link_age = _s_ago(latest.received_at, now)
    link_alive = link_age <= LINK_MAX_AGE_S

    if not link_alive:
        return SensorHealthReport(
            link_alive=False,
            sample_rate_hz=link.sample_rate_hz(),
            per_sensor={
                "link": SensorState("link", "failed",
                                     f"no frames for {link_age:.1f} s (>{LINK_MAX_AGE_S:.0f} s)",
                                     last_seen_s_ago=link_age,
                                     fault_id="sensor_loss_link"),
            },
            active_loss_events=["sensor_loss_link"],
        )

    per: dict[str, SensorState] = {}
    events: list[str] = []

    # ── IMU / gyro health ──────────────────────────────────────────────
    imu_state = _evaluate_imu(samples, now)
    per["imu"] = imu_state
    if imu_state.fault_id:
        events.append(imu_state.fault_id)

    # ── Ultrasonic range health ────────────────────────────────────────
    range_state = _evaluate_range(samples, now)
    per["range"] = range_state
    if range_state.fault_id:
        events.append(range_state.fault_id)

    # ── Temperature (DS18B20 or on-die) ────────────────────────────────
    temp_state = _evaluate_temp(samples)
    per["temp"] = temp_state
    if temp_state.fault_id:
        events.append(temp_state.fault_id)

    # ── Overall link (rate degradation) ────────────────────────────────
    rate = link.sample_rate_hz()
    if rate < LOW_RATE_HZ:
        per["link"] = SensorState("link", "degraded",
                                   f"sample rate {rate:.1f} Hz below {LOW_RATE_HZ:.0f} Hz",
                                   last_seen_s_ago=link_age)
    else:
        per["link"] = _nominal("link", link_age)

    return SensorHealthReport(
        link_alive=True,
        sample_rate_hz=rate,
        per_sensor=per,
        active_loss_events=events,
    )


def _evaluate_imu(samples, now: float) -> SensorState:
    """The IMU is our safety-critical sensor; check for stale and flatline."""
    latest = samples[-1]

    # Out-of-range check on the latest sample.
    if (abs(latest.gyro_x) > IMU_MAX_GYRO_DPS or
        abs(latest.gyro_y) > IMU_MAX_GYRO_DPS or
        abs(latest.gyro_z) > IMU_MAX_GYRO_DPS or
        abs(latest.accel_x) > IMU_MAX_ACCEL_G or
        abs(latest.accel_y) > IMU_MAX_ACCEL_G or
        abs(latest.accel_z) > IMU_MAX_ACCEL_G):
        return SensorState("imu", "failed",
                           "readings beyond hardware maximum — sensor fault",
                           fault_id="sensor_loss_imu")

    # Flatline: exact zeros across all three gyro axes for many frames in a
    # row. The MPU6050 never sits at true 0.0 on all three simultaneously
    # under noise — this pattern only appears when the I2C read returns
    # nothing and the ints get cast through 0.
    flat_run = 0
    for sample in reversed(samples):
        if (abs(sample.gyro_x) < IMU_FLATLINE_EPS and
            abs(sample.gyro_y) < IMU_FLATLINE_EPS and
            abs(sample.gyro_z) < IMU_FLATLINE_EPS):
            flat_run += 1
            if flat_run >= IMU_FLATLINE_COUNT:
                return SensorState("imu", "failed",
                                   f"gyro flatline for {flat_run} consecutive frames "
                                   f"(I2C failure suspected)",
                                   fault_id="sensor_loss_imu")
        else:
            break

    # Stale: the exact same gyro triple repeated for more than STALE_TOL
    # seconds. A live IMU has quantisation noise; a frozen one doesn't.
    stale_since = None
    reference = (latest.gyro_x, latest.gyro_y, latest.gyro_z)
    for sample in reversed(samples):
        if (sample.gyro_x, sample.gyro_y, sample.gyro_z) == reference:
            stale_since = sample.received_at
        else:
            break
    stale_span = _s_ago(stale_since, now) if stale_since is not None else 0.0
    if stale_span >= IMU_STALE_SECONDS:
        return SensorState("imu", "failed",
                           f"identical gyro triple for {stale_span:.1f} s (>{IMU_STALE_SECONDS:.0f} s) "
                           f"— sensor frozen",
                           fault_id="sensor_loss_imu")

    return _nominal("imu", 0.0)


def _evaluate_range(samples, now: float) -> SensorState:
    """The HC-SR04 reports None on timeout — sustained None = failed transducer."""
    latest = samples[-1]

    # If the latest sample has a distance, we're healthy.
    if latest.distance_cm is not None and latest.distance_cm >= 0:
        return _nominal("range", 0.0)

    # Otherwise walk backwards looking for the last valid echo. If we've gone
    # >5 s without one, treat as failed.
    last_valid = None
    for sample in reversed(samples):
        if sample.distance_cm is not None and sample.distance_cm >= 0:
            last_valid = sample.received_at
            break
    span = _s_ago(last_valid, now) if last_valid is not None else \
           _s_ago(samples[0].received_at, now)
    if span >= RANGE_NO_ECHO_SECONDS:
        return SensorState("range", "failed",
                           f"no valid echo for {span:.1f} s (>{RANGE_NO_ECHO_SECONDS:.0f} s)",
                           last_seen_s_ago=span,
                           fault_id="sensor_loss_range")
    return SensorState("range", "degraded",
                       f"no echo on latest frame (last valid {span:.1f} s ago)",
                       last_seen_s_ago=span)


def _evaluate_temp(samples) -> SensorState:
    """Temperature isn't safety-critical but flag it if consistently absent."""
    latest = samples[-1]
    if latest.temp_c is None:
        return SensorState("temp", "degraded",
                           "no temperature reading (DS18B20 or on-die probe absent)")
    # A wildly wrong reading (below -40 or above 85 C) is almost certainly a
    # bus fault. Space-hardware temperatures on this board should never be
    # outside 0-60 C during a demo.
    if latest.temp_c < -40 or latest.temp_c > 100:
        return SensorState("temp", "failed",
                           f"reading {latest.temp_c:.1f} °C outside plausible range",
                           fault_id="sensor_loss_temp")
    return _nominal("temp", 0.0)


# ── Serialisation for the tier prompts ──────────────────────────────────────

def health_to_payload(report: SensorHealthReport) -> dict:
    """Compact dict for inclusion in the featurised state Gemma sees.

    Only informative fields; keeps token cost tiny.
    """
    return {
        "link_alive": report.link_alive,
        "sample_rate_hz": round(report.sample_rate_hz, 2),
        "summary": report.summary(),
        "active_loss_events": list(report.active_loss_events),
        "per_sensor": {
            name: {"status": state.status, "reason": state.reason}
            for name, state in report.per_sensor.items()
        },
    }
