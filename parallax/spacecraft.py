import copy
import re
import numpy as np
from dataclasses import dataclass, field


HEALTH_STATES = {"nominal", "degraded", "failed", "recovering"}
SEVERITY_TO_HEALTH = {"advisory": "degraded", "warning": "degraded", "critical": "failed"}

# Ordering used when several concurrent faults touch the same subsystem or
# component: the worst reported health always wins, never the last one applied.
HEALTH_SEVERITY = {"nominal": 0, "recovering": 1, "degraded": 2, "failed": 3}

# These names are shared with the 3D model.  A custom event can target any of
# them without needing a new hard-coded fault definition.
VISUAL_COMPONENTS = [
    "Solar array +X",
    "Solar array -X",
    "Power conditioning unit",
    "Radiator panels",
    "Reaction-wheel assembly",
    "High-gain antenna",
    "Low-gain antenna",
    "Spectrometer",
    "Camera",
    "Thrusters",
]

TELEMETRY_FIELDS = {
    "solar_output_w", "solar_efficiency_pct", "battery_soc_pct", "pcu_temp_c",
    "bus_voltage_v", "power_draw_w", "bus_temp_c", "instrument_temp_c",
    "radiator_active", "attitude_error_arcsec", "rw_speed_rpm",
    "sun_pointing_error_deg", "adcs_mode", "signal_strength_dbm", "link_margin_db",
    "data_rate_mbps", "antenna_mode", "spectrometer_status", "spectrometer_output",
    "camera_status", "radiation_cps", "tank_pressure_bar", "fuel_mass_kg",
    "thruster_status",
}


@dataclass
class SpacecraftState:
    # Mission
    mission_time_s: float = 0.0

    # Power subsystem
    solar_output_w: float = 850.0
    solar_efficiency_pct: float = 100.0
    battery_soc_pct: float = 94.2
    pcu_temp_c: float = 41.3
    bus_voltage_v: float = 28.1
    power_draw_w: float = 420.0

    # Thermal subsystem
    bus_temp_c: float = 22.1
    instrument_temp_c: float = 18.0
    radiator_active: bool = True

    # ADCS subsystem
    attitude_error_arcsec: float = 0.3
    rw_speed_rpm: float = 2801.0
    sun_pointing_error_deg: float = 0.008
    adcs_mode: str = "fine_pointing"

    # Comms subsystem
    signal_strength_dbm: float = -114.8
    link_margin_db: float = 8.2
    data_rate_mbps: float = 1.2
    antenna_mode: str = "hga"

    # Science instruments
    spectrometer_status: str = "active"
    spectrometer_output: float = 10.2
    camera_status: str = "standby"
    radiation_cps: float = 31.2

    # Propulsion
    tank_pressure_bar: float = 240.1
    fuel_mass_kg: float = 45.2
    thruster_status: str = "nominal"

    # Fault state
    active_faults: list = field(default_factory=list)
    fault_metadata: dict = field(default_factory=dict)
    component_states: dict = field(default_factory=dict)
    subsystem_health: dict = field(default_factory=lambda: {
        "Power": "nominal",
        "Thermal": "nominal",
        "ADCS": "nominal",
        "Communications": "nominal",
        "Science": "nominal",
        "Propulsion": "nominal",
    })


def copy_state(state: SpacecraftState) -> SpacecraftState:
    """Return a deep copy so UI previews cannot mutate the live spacecraft."""
    return copy.deepcopy(state)


def add_noise(state: SpacecraftState, seed: int = 0) -> SpacecraftState:
    """Add small realistic noise to sensor readings."""
    rng = np.random.default_rng(seed)
    state.solar_output_w += rng.normal(0, 3)
    state.battery_soc_pct += rng.normal(0, 0.05)
    state.pcu_temp_c += rng.normal(0, 0.2)
    state.bus_voltage_v += rng.normal(0, 0.05)
    state.bus_temp_c += rng.normal(0, 0.1)
    state.instrument_temp_c += rng.normal(0, 0.05)
    state.attitude_error_arcsec += abs(rng.normal(0, 0.05))
    state.rw_speed_rpm += rng.normal(0, 5)
    state.signal_strength_dbm += rng.normal(0, 0.3)
    state.link_margin_db += rng.normal(0, 0.1)
    state.radiation_cps += rng.normal(0, 0.5)
    if state.spectrometer_output is not None and state.spectrometer_status == "active":
        state.spectrometer_output += rng.normal(0, 0.15)
    return state


FAULT_DEFINITIONS = {
    "solar_string_loss": {
        "label": "Solar String Failure",
        "icon": "☀",
        "subsystem": "Power",
        "description": "Solar array string 2 open-circuit fault. Output drops 35%.",
        "apply": lambda s: _apply_solar_string_loss(s),
    },
    "pcu_fault": {
        "label": "PCU Electronics Fault",
        "icon": "⚡",
        "subsystem": "Power",
        "description": "Power conditioning unit component failure. Bus voltage unstable.",
        "apply": lambda s: _apply_pcu_fault(s),
    },
    "reaction_wheel_fault": {
        "label": "Reaction Wheel Bearing",
        "icon": "🌀",
        "subsystem": "ADCS",
        "description": "Reaction wheel 2 bearing degradation. ADCS pointing compromised.",
        "apply": lambda s: _apply_rw_fault(s),
    },
    "spectrometer_fault": {
        "label": "Spectrometer Corruption",
        "icon": "🔭",
        "subsystem": "Science",
        "description": "Science instrument producing corrupted data. Instrument offline.",
        "apply": lambda s: _apply_spectrometer_fault(s),
    },
    "comms_dropout": {
        "label": "Comms Dropout",
        "icon": "📡",
        "subsystem": "Communications",
        "description": "High-gain antenna misalignment. Signal below threshold.",
        "apply": lambda s: _apply_comms_dropout(s),
    },
    "thermal_runaway": {
        "label": "Thermal Runaway",
        "icon": "🌡",
        "subsystem": "Thermal",
        "description": "PCU thermal spike. Cascading risk to power and electronics.",
        "apply": lambda s: _apply_thermal_runaway(s),
    },
}


def inject_fault(state: SpacecraftState, fault_id: str) -> SpacecraftState:
    """Add one fault on top of whatever is already broken.

    Injection is idempotent: re-triggering an active fault must not stack its
    telemetry damage a second time.
    """
    if fault_id in FAULT_DEFINITIONS and fault_id not in state.active_faults:
        state = FAULT_DEFINITIONS[fault_id]["apply"](state)
        state.active_faults.append(fault_id)
    return state


def clear_fault(state: SpacecraftState, fault_id: str) -> SpacecraftState:
    """Remove one fault, keeping every other active fault intact.

    Telemetry damage is not individually reversible, so the state is rebuilt
    from a clean spacecraft with the remaining faults re-applied.
    """
    remaining = [f for f in state.active_faults if f != fault_id]
    metadata = {k: v for k, v in state.fault_metadata.items() if k != fault_id}
    return rebuild_state(remaining, metadata)


def rebuild_state(fault_ids: list[str], fault_metadata: dict | None = None) -> SpacecraftState:
    """Build a spacecraft state carrying exactly `fault_ids`, applied in order."""
    state = SpacecraftState()
    state.fault_metadata = copy.deepcopy(fault_metadata or {})
    for fault_id in fault_ids:
        if fault_id in FAULT_DEFINITIONS:
            state = FAULT_DEFINITIONS[fault_id]["apply"](state)
        elif fault_id in state.fault_metadata:
            _apply_custom_metadata(state, state.fault_metadata[fault_id])
        else:
            continue
        state.active_faults.append(fault_id)
    return state


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value or "custom_anomaly"


def _validate_telemetry_overrides(overrides: dict) -> dict:
    if not isinstance(overrides, dict):
        raise ValueError("Telemetry overrides must be a JSON object.")

    unknown = set(overrides) - TELEMETRY_FIELDS
    if unknown:
        raise ValueError(f"Unsupported telemetry field(s): {', '.join(sorted(unknown))}.")

    validated = {}
    for name, value in overrides.items():
        if name == "spectrometer_output" and value is None:
            validated[name] = None
        elif name == "radiator_active":
            if not isinstance(value, bool):
                raise ValueError("radiator_active must be true or false.")
            validated[name] = value
        elif name in {"adcs_mode", "antenna_mode", "spectrometer_status", "camera_status", "thruster_status"}:
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string.")
            validated[name] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            validated[name] = float(value)
        else:
            raise ValueError(f"{name} must be a number.")
    return validated


def inject_custom_fault(
    state: SpacecraftState,
    *,
    title: str,
    description: str,
    subsystems: list[str],
    severity: str,
    telemetry_overrides: dict | None = None,
    components: list[str] | None = None,
) -> tuple[SpacecraftState, str]:
    """Apply a user-defined anomaly through validated, data-driven state changes.

    This deliberately does not infer numerical changes from prose.  The
    operator supplies optional telemetry overrides, while the free-text event
    description is passed to Gemma as context for reasoning.
    """
    title = title.strip()
    description = description.strip()
    if not title:
        raise ValueError("Give the anomaly a short title.")
    if not description:
        raise ValueError("Describe the observed condition for Gemma.")
    if not subsystems:
        raise ValueError("Select at least one affected subsystem.")
    if severity not in SEVERITY_TO_HEALTH:
        raise ValueError("Severity must be advisory, warning, or critical.")

    invalid_subsystems = set(subsystems) - set(state.subsystem_health)
    if invalid_subsystems:
        raise ValueError(f"Unknown subsystem(s): {', '.join(sorted(invalid_subsystems))}.")
    components = components or []
    invalid_components = set(components) - set(VISUAL_COMPONENTS)
    if invalid_components:
        raise ValueError(f"Unknown component(s): {', '.join(sorted(invalid_components))}.")

    overrides = _validate_telemetry_overrides(telemetry_overrides or {})
    base_id = _slug(title)
    suffix = 1
    fault_id = base_id
    while fault_id in state.fault_metadata or fault_id in FAULT_DEFINITIONS:
        suffix += 1
        fault_id = f"{base_id}_{suffix}"

    state.fault_metadata[fault_id] = {
        "label": title,
        "icon": "✦",
        "description": description,
        "subsystems": subsystems,
        "severity": severity,
        "telemetry_overrides": overrides,
        "components": components,
    }
    _apply_custom_metadata(state, state.fault_metadata[fault_id])
    state.active_faults.append(fault_id)
    return state, fault_id


def _apply_custom_metadata(state: SpacecraftState, metadata: dict) -> None:
    """Apply a stored custom anomaly, alongside any other active faults."""
    for field_name, value in metadata.get("telemetry_overrides", {}).items():
        setattr(state, field_name, value)

    health = SEVERITY_TO_HEALTH[metadata.get("severity", "warning")]
    for subsystem in metadata.get("subsystems", []):
        _degrade_health(state, subsystem, health)
    for component in metadata.get("components", []):
        _degrade_component(state, component, health)


def fault_details(state: SpacecraftState, fault_id: str) -> dict:
    """Return a presentation-safe definition for preset and custom events."""
    return state.fault_metadata.get(fault_id, FAULT_DEFINITIONS.get(fault_id, {
        "label": fault_id.replace("_", " ").title(), "icon": "⚠", "description": "",
    }))


# Faults can be active at the same time, so every mutation below is written to
# compose: rates scale multiplicatively, absolute readings settle on the worse
# of the two values, and health only ever degrades further.

def _degrade_health(s: SpacecraftState, subsystem: str, health: str) -> None:
    current = s.subsystem_health.get(subsystem, "nominal")
    if HEALTH_SEVERITY[health] > HEALTH_SEVERITY.get(current, 0):
        s.subsystem_health[subsystem] = health


def _degrade_component(s: SpacecraftState, component: str, health: str) -> None:
    current = s.component_states.get(component, "nominal")
    if HEALTH_SEVERITY[health] > HEALTH_SEVERITY.get(current, 0):
        s.component_states[component] = health


def _worse_high(current: float, value: float) -> float:
    """For readings where higher is worse (temperatures, pointing error)."""
    return max(current, value)


def _worse_low(current: float, value: float) -> float:
    """For readings where lower is worse (voltage, link margin, data rate)."""
    return min(current, value)


def _apply_solar_string_loss(s: SpacecraftState) -> SpacecraftState:
    s.solar_output_w *= 0.65
    s.solar_efficiency_pct = _worse_low(s.solar_efficiency_pct, 65.0)
    s.battery_soc_pct -= 2.1
    _degrade_health(s, "Power", "degraded")
    return s

def _apply_pcu_fault(s: SpacecraftState) -> SpacecraftState:
    s.bus_voltage_v = _worse_low(s.bus_voltage_v, 24.8)
    s.pcu_temp_c = _worse_high(s.pcu_temp_c, 58.4)
    s.power_draw_w *= 1.15
    _degrade_health(s, "Power", "failed")
    _degrade_health(s, "Thermal", "degraded")
    return s

def _apply_rw_fault(s: SpacecraftState) -> SpacecraftState:
    s.rw_speed_rpm = _worse_low(s.rw_speed_rpm, 380.0)
    s.attitude_error_arcsec = _worse_high(s.attitude_error_arcsec, 7.8)
    s.sun_pointing_error_deg = _worse_high(s.sun_pointing_error_deg, 0.42)
    s.adcs_mode = "degraded"
    _degrade_health(s, "ADCS", "failed")
    s.solar_output_w *= 0.88  # off-sun pointing reduces solar
    return s

def _apply_spectrometer_fault(s: SpacecraftState) -> SpacecraftState:
    s.spectrometer_status = "failed"
    s.spectrometer_output = None
    _degrade_health(s, "Science", "failed")
    return s

def _apply_comms_dropout(s: SpacecraftState) -> SpacecraftState:
    s.signal_strength_dbm = _worse_low(s.signal_strength_dbm, -124.6)
    s.link_margin_db = _worse_low(s.link_margin_db, -1.4)
    s.data_rate_mbps = _worse_low(s.data_rate_mbps, 0.115)
    s.antenna_mode = "lga_fallback"
    _degrade_health(s, "Communications", "failed")
    return s

def _apply_thermal_runaway(s: SpacecraftState) -> SpacecraftState:
    s.pcu_temp_c = _worse_high(s.pcu_temp_c, 71.2)
    s.bus_temp_c = _worse_high(s.bus_temp_c, 38.6)
    s.instrument_temp_c = _worse_high(s.instrument_temp_c, 29.1)
    _degrade_health(s, "Thermal", "failed")
    _degrade_health(s, "Power", "degraded")
    s.solar_output_w *= 0.82
    s.bus_voltage_v = _worse_low(s.bus_voltage_v, 26.2)
    return s
