"""Live sensor link to the Arduino/ESP32 handle over USB serial.

The board streams gyro, accelerometer, external temperature and ultrasonic
range at ~5 Hz.  Reading happens on a background thread so Streamlit reruns
never block on the serial port; the UI only ever touches the sample buffer.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:  # pyserial is optional — the rest of the dashboard still runs
    SERIAL_AVAILABLE = False


BAUD_RATE = 460800
BUFFER_SAMPLES = 3000         # ~60 s of history at 50 Hz

# Sensor sentinels emitted by the sketch when a peripheral does not answer.
TEMP_DISCONNECTED = -127.0    # DallasTemperature: no DS18B20 on the bus
DISTANCE_NO_ECHO = -1.0       # HC-SR04: pulseIn timed out


@dataclass
class SensorSample:
    """One decoded line of board telemetry. Absent sensors are None, not sentinels."""
    received_at: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    accel_x: float
    accel_y: float
    accel_z: float
    temp_c: float | None = None
    distance_cm: float | None = None
    # The board's own millis() when the frame was built, when the firmware
    # sends it. Integrating on this clock avoids USB arrival jitter.
    board_ms: float | None = None
    # Filled in by OrientationFilter as frames arrive, in degrees.
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @property
    def gyro_magnitude(self) -> float:
        """Total body rate in deg/s."""
        return math.sqrt(self.gyro_x ** 2 + self.gyro_y ** 2 + self.gyro_z ** 2)

    @property
    def accel_magnitude(self) -> float:
        """Total specific force in g — about 1.0 when the board is at rest."""
        return math.sqrt(self.accel_x ** 2 + self.accel_y ** 2 + self.accel_z ** 2)

    @property
    def tilt_deg(self) -> float:
        """Angle between the board's Z axis and the gravity vector."""
        horizontal = math.sqrt(self.accel_x ** 2 + self.accel_y ** 2)
        return math.degrees(math.atan2(horizontal, self.accel_z))


def _clean(value: float, sentinel: float) -> float | None:
    return None if abs(value - sentinel) < 0.05 else value


def parse_line(line: str) -> SensorSample | None:
    """Decode one serial line, or return None if it is not a telemetry frame.

    Both firmware output formats are accepted, so the dashboard works whether
    the board is running the CSV sketch or the human-readable one:
      DATA,0.18,-0.07,-0.03,-0.10,-0.02,1.07,-127.0,-1.0
      Gyro X: 0.18 | Gyro Y: -0.07 | ... | Temp: -127.0 C | Dist: -1.0 cm
    """
    line = line.strip()
    if not line:
        return None

    values: list[float] = []
    if line.startswith("DATA,"):
        try:
            values = [float(part) for part in line[len("DATA,"):].split(",")]
        except ValueError:
            return None
    elif "Gyro X" in line:
        for field_text in line.split("|"):
            _, _, raw = field_text.partition(":")
            raw = raw.strip().rstrip("Ccm").strip()
            try:
                values.append(float(raw))
            except ValueError:
                return None
    else:
        return None

    if len(values) < 6:
        return None

    return SensorSample(
        received_at=time.time(),
        gyro_x=values[0], gyro_y=values[1], gyro_z=values[2],
        accel_x=values[3], accel_y=values[4], accel_z=values[5],
        temp_c=_clean(values[6], TEMP_DISCONNECTED) if len(values) > 6 else None,
        distance_cm=_clean(values[7], DISTANCE_NO_ECHO) if len(values) > 7 else None,
        board_ms=values[8] if len(values) > 8 else None,
    )


class OrientationFilter:
    """Complementary filter turning gyro rates + gravity into a board attitude.

    Roll and pitch are held to the accelerometer's gravity vector so they do
    not drift.  Yaw has no absolute reference on this board — there is no
    magnetometer — so it is pure gyro integration and will creep over time.
    """

    # Time constant of the pull toward gravity, in seconds. The per-sample blend
    # weight is derived from it and dt, so the filter behaves identically at
    # 5 Hz and 50 Hz — a fixed weight would be 10x more aggressive at 50 Hz.
    CORRECTION_TAU_S = 1.5
    # Only trust gravity when the board is not being accelerated (|a| near 1 g).
    STILL_TOLERANCE_G = 0.35
    MAX_DT_S = 0.5   # after a gap, restart integration rather than lurching

    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self._last_t: float | None = None

    def reset(self) -> None:
        self.roll = self.pitch = self.yaw = 0.0
        self._last_t = None

    def update(self, sample: SensorSample) -> None:
        # Prefer the board's clock: USB delivery jitter is comparable to the
        # frame interval at 50 Hz, which would show up as integration noise.
        now = sample.board_ms / 1000.0 if sample.board_ms is not None else sample.received_at
        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now
        if dt <= 0 or dt > self.MAX_DT_S:
            dt = 0.0

        # Integrate body rates (deg/s) forward.
        roll = self.roll + sample.gyro_x * dt
        pitch = self.pitch + sample.gyro_y * dt
        yaw = self.yaw + sample.gyro_z * dt

        # Correct roll/pitch against gravity when the board is roughly at rest.
        if abs(sample.accel_magnitude - 1.0) < self.STILL_TOLERANCE_G:
            horizontal = math.sqrt(sample.accel_y ** 2 + sample.accel_z ** 2)
            roll_ref = math.degrees(math.atan2(sample.accel_y, sample.accel_z))
            pitch_ref = math.degrees(math.atan2(-sample.accel_x, horizontal))
            blend = self.CORRECTION_TAU_S / (self.CORRECTION_TAU_S + dt) if dt > 0 else 0.0
            roll = blend * roll + (1 - blend) * roll_ref
            pitch = blend * pitch + (1 - blend) * pitch_ref

        self.roll = _wrap_deg(roll)
        self.pitch = _wrap_deg(pitch)
        self.yaw = _wrap_deg(yaw)

        sample.roll, sample.pitch, sample.yaw = self.roll, self.pitch, self.yaw


def _wrap_deg(angle: float) -> float:
    """Fold an angle into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def list_ports() -> list[tuple[str, str]]:
    """Available serial devices as (device, description), USB adapters first."""
    if not SERIAL_AVAILABLE:
        return []
    ports = [(p.device, p.description or "serial device") for p in serial.tools.list_ports.comports()]
    ports.sort(key=lambda p: ("usbserial" not in p[0] and "usbmodem" not in p[0], p[0]))
    return ports


class SerialLink:
    """Background reader holding a rolling buffer of the newest samples."""

    def __init__(self, buffer_samples: int = BUFFER_SAMPLES):
        self._lock = threading.Lock()
        self._samples: deque[SensorSample] = deque(maxlen=buffer_samples)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._orientation = OrientationFilter()
        self.port: str | None = None
        self.error: str | None = None
        self.board_message: str | None = None
        self.connected = False
        self.lines_seen = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self, port: str, baud: int = BAUD_RATE) -> None:
        if self.is_running() and self.port == port:
            return
        self.stop()

        self._stop.clear()
        self.port = port
        self.error = None
        self.board_message = None
        self.lines_seen = 0
        self._orientation.reset()
        with self._lock:
            self._samples.clear()

        self._thread = threading.Thread(target=self._read_loop, args=(port, baud), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self.connected = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── reader thread ────────────────────────────────────────────────────────

    def _read_loop(self, port: str, baud: int) -> None:
        try:
            connection = serial.Serial(port, baud, timeout=1.0)
        except Exception as exc:
            self.error = _friendly_error(exc, port)
            self.connected = False
            return

        self.connected = True
        try:
            # The ESP32 reboots when the port opens, so drop the boot chatter.
            time.sleep(0.3)
            connection.reset_input_buffer()

            while not self._stop.is_set():
                try:
                    raw = connection.readline().decode("utf-8", errors="replace")
                except Exception as exc:
                    self.error = _friendly_error(exc, port)
                    break
                if not raw.strip():
                    continue

                self.lines_seen += 1
                sample = parse_line(raw)
                if sample is not None:
                    self._orientation.update(sample)
                    with self._lock:
                        self._samples.append(sample)
                else:
                    # Status text such as "Calibrating... keep still".
                    self.board_message = raw.strip()[:120]
        finally:
            self.connected = False
            try:
                connection.close()
            except Exception:
                pass

    # ── readers ──────────────────────────────────────────────────────────────

    def reset_orientation(self) -> None:
        """Re-zero the attitude estimate — mainly to clear accumulated yaw drift."""
        self._orientation.reset()

    def latest(self) -> SensorSample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def history(self) -> list[SensorSample]:
        with self._lock:
            return list(self._samples)

    def sample_rate_hz(self) -> float:
        """Observed frame rate over the buffer — the sketch targets 5 Hz."""
        with self._lock:
            if len(self._samples) < 2:
                return 0.0
            span = self._samples[-1].received_at - self._samples[0].received_at
            return (len(self._samples) - 1) / span if span > 0 else 0.0

    def is_live(self, max_age_s: float = 2.0) -> bool:
        sample = self.latest()
        return sample is not None and (time.time() - sample.received_at) <= max_age_s


def _friendly_error(exc: Exception, port: str) -> str:
    text = str(exc)
    if "Resource busy" in text or "Errno 16" in text:
        return (f"{port} is held by another program. Close the Arduino IDE "
                f"Serial Monitor (or Serial Plotter) and reconnect.")
    if "No such file" in text or "could not open" in text.lower():
        return f"{port} is not available. Check the USB cable and the selected port."
    return text


# ── mapping the handle onto spacecraft telemetry ─────────────────────────────

# Body rate and tilt drive the ADCS readings; the DS18B20 drives instrument
# temperature.  Scale factors are chosen so ordinary desk handling spans the
# dashboard's nominal-to-alarm range.
ARCSEC_PER_DEG_PER_S = 15.0
NOMINAL_ATTITUDE_ARCSEC = 0.3


def map_to_spacecraft(sample: SensorSample) -> dict:
    """Translate one board sample into spacecraft telemetry fields."""
    mapped = {
        "attitude_error_arcsec": NOMINAL_ATTITUDE_ARCSEC + sample.gyro_magnitude * ARCSEC_PER_DEG_PER_S,
        "sun_pointing_error_deg": round(sample.tilt_deg, 3),
    }
    if sample.temp_c is not None:
        mapped["instrument_temp_c"] = sample.temp_c
    return mapped


def apply_to_state(state, sample: SensorSample) -> None:
    """Write the mapped hardware readings onto a live spacecraft state."""
    for name, value in map_to_spacecraft(sample).items():
        setattr(state, name, value)
