"""Whitelisted recovery-plan action vocabulary.

Gemma never talks to the spacecraft. It talks to this file. G2 emits a plan
whose every step must resolve to one of the ACTIONS defined here, and the
plan then flows through the validator gates in `validator.py` before any
mutation is applied to state. Adding a new capability is a code change, not a
prompt change — that is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from parallax.spacecraft import SpacecraftState, copy_state


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    # A short human phrase used in the UI to explain what this step will do.
    human_label: str
    # (param_name, param_type, allowed_values_or_range) — checked in the validator
    # before the effect runs. `range` is a (lo, hi) tuple; `choices` is a set.
    params: dict[str, dict[str, Any]]
    # Cost model: (subsystem impact used by the safety envelope check).
    # `battery_delta_pct` is the *expected* battery cost of executing this step
    # in the near term. Positive means it drains, negative means it recovers.
    battery_delta_pct: float
    # Whether this step requires the named comms path to stay alive after it
    # completes ("hga", "lga", or None for "no comms constraint").
    requires_comms_path: str | None
    # Deterministic effect on state. Called during apply(); must accept
    # (state, params) and return a mutated state.
    apply_fn: Callable[[SpacecraftState, dict], SpacecraftState]


# ── Concrete effects (act on a copy of state) ────────────────────────────────

def _shed_load(state: SpacecraftState, params: dict) -> SpacecraftState:
    pct = float(params["pct"])
    state.power_draw_w *= max(0.0, 1.0 - pct / 100.0)
    # A brief thermal reprieve — reducing draw reduces PCU dissipation.
    state.pcu_temp_c = max(30.0, state.pcu_temp_c - pct * 0.15)
    # Battery relief comes from reducing draw, not from generating more energy.
    state.battery_soc_pct = min(100.0, state.battery_soc_pct + pct * 0.05)
    return state


def _switch_pcu(state: SpacecraftState, params: dict) -> SpacecraftState:
    if params.get("target", "pcu_a") == "pcu_a":
        # The backup restores the bus voltage but the switchover blip costs
        # some battery. This is what makes the plan look real to the validator.
        state.bus_voltage_v = max(state.bus_voltage_v, 27.9)
        state.pcu_temp_c = min(state.pcu_temp_c, 48.0)
        state.battery_soc_pct -= 0.4
        state.subsystem_health["Power"] = "recovering"
    return state


def _slew_to_sun(state: SpacecraftState, params: dict) -> SpacecraftState:
    state.sun_pointing_error_deg = min(state.sun_pointing_error_deg, 0.02)
    state.solar_output_w *= 1.10
    state.attitude_error_arcsec = min(state.attitude_error_arcsec, 2.5)
    state.adcs_mode = "sun_pointing"
    return state


def _enter_safe_mode(state: SpacecraftState, params: dict) -> SpacecraftState:
    # Safe mode: minimal draw, sun-pointing, science quiescent.
    state.power_draw_w = min(state.power_draw_w, 240.0)
    state.adcs_mode = "safe_mode"
    state.spectrometer_status = "safe"
    state.camera_status = "off"
    state.sun_pointing_error_deg = min(state.sun_pointing_error_deg, 0.05)
    for subsystem in state.subsystem_health:
        if state.subsystem_health[subsystem] == "failed":
            state.subsystem_health[subsystem] = "recovering"
    return state


def _suspend_science(state: SpacecraftState, params: dict) -> SpacecraftState:
    if state.spectrometer_status != "failed":
        state.spectrometer_status = "safe"
    state.camera_status = "off"
    state.power_draw_w = max(0.0, state.power_draw_w - 60.0)
    return state


def _switch_antenna(state: SpacecraftState, params: dict) -> SpacecraftState:
    target = params.get("target", "lga")
    if target == "lga":
        state.antenna_mode = "lga_fallback"
        state.data_rate_mbps = max(state.data_rate_mbps, 0.115)
        state.signal_strength_dbm = max(state.signal_strength_dbm, -118.4)
        state.link_margin_db = max(state.link_margin_db, 1.6)
        if state.subsystem_health.get("Communications") == "failed":
            state.subsystem_health["Communications"] = "recovering"
    else:
        state.antenna_mode = "hga"
    return state


def _activate_backup_rw(state: SpacecraftState, params: dict) -> SpacecraftState:
    state.rw_speed_rpm = max(state.rw_speed_rpm, 1800.0)
    state.attitude_error_arcsec = min(state.attitude_error_arcsec, 3.0)
    state.adcs_mode = "torquer_backup"
    if state.subsystem_health.get("ADCS") == "failed":
        state.subsystem_health["ADCS"] = "recovering"
    return state


def _deploy_radiators(state: SpacecraftState, params: dict) -> SpacecraftState:
    state.radiator_active = True
    state.pcu_temp_c = max(30.0, state.pcu_temp_c - 6.5)
    state.bus_temp_c = max(18.0, state.bus_temp_c - 2.4)
    state.instrument_temp_c = max(14.0, state.instrument_temp_c - 1.1)
    if state.subsystem_health.get("Thermal") == "failed":
        state.subsystem_health["Thermal"] = "recovering"
    return state


def _reduce_compute_load(state: SpacecraftState, params: dict) -> SpacecraftState:
    pct = float(params["pct"])
    state.pcu_temp_c = max(30.0, state.pcu_temp_c - pct * 0.20)
    state.power_draw_w *= max(0.0, 1.0 - pct / 200.0)
    return state


def _bypass_solar_string(state: SpacecraftState, params: dict) -> SpacecraftState:
    state.solar_output_w *= 1.28
    state.solar_efficiency_pct = min(100.0, state.solar_efficiency_pct + 20.0)
    if state.subsystem_health.get("Power") in {"failed", "degraded"}:
        state.subsystem_health["Power"] = "recovering"
    return state


def _hold_and_observe(state: SpacecraftState, params: dict) -> SpacecraftState:
    # Explicit "do nothing risky, watch the telemetry" step. Present so
    # G2 can build a plan that observes for N seconds before executing —
    # useful when the diagnostician's confidence is low.
    return state


# ── The vocabulary ───────────────────────────────────────────────────────────

ACTIONS: dict[str, ActionSpec] = {
    "shed_load": ActionSpec(
        name="shed_load",
        description="Reduce non-essential loads by the given percentage.",
        human_label="Shed non-essential load",
        params={"pct": {"type": "number", "range": (5, 60)}},
        battery_delta_pct=-1.5,   # small drain in the moment, big gain later
        requires_comms_path=None,
        apply_fn=_shed_load,
    ),
    "switch_pcu": ActionSpec(
        name="switch_pcu",
        description="Switch to the redundant power conditioning unit.",
        human_label="Switch to backup PCU",
        params={"target": {"type": "string", "choices": {"pcu_a"}}},
        battery_delta_pct=-0.4,
        requires_comms_path=None,
        apply_fn=_switch_pcu,
    ),
    "slew_to_sun": ActionSpec(
        name="slew_to_sun",
        description="Point the +Z axis toward the Sun to maximise solar input.",
        human_label="Slew to sun",
        params={},
        battery_delta_pct=-0.2,
        requires_comms_path=None,
        apply_fn=_slew_to_sun,
    ),
    "enter_safe_mode": ActionSpec(
        name="enter_safe_mode",
        description="Enter spacecraft safe mode: minimal draw, Sun-pointing, science quiet.",
        human_label="Enter safe mode",
        params={},
        battery_delta_pct=+3.0,
        requires_comms_path=None,
        apply_fn=_enter_safe_mode,
    ),
    "suspend_science": ActionSpec(
        name="suspend_science",
        description="Quiesce the science instruments to preserve power and data.",
        human_label="Suspend science operations",
        params={},
        battery_delta_pct=+1.2,
        requires_comms_path=None,
        apply_fn=_suspend_science,
    ),
    "switch_antenna": ActionSpec(
        name="switch_antenna",
        description="Fall back to LGA or return to HGA.",
        human_label="Switch antenna",
        params={"target": {"type": "string", "choices": {"lga", "hga"}}},
        battery_delta_pct=-0.1,
        # The whole point of this action is to *change* the comms path, so it
        # is exempt from the requires-comms check — the validator only guards
        # steps that *rely* on a specific path.
        requires_comms_path=None,
        apply_fn=_switch_antenna,
    ),
    "activate_backup_rw": ActionSpec(
        name="activate_backup_rw",
        description="Engage magnetic torquers or the backup wheel to hold attitude.",
        human_label="Activate backup attitude control",
        params={},
        battery_delta_pct=-0.5,
        requires_comms_path=None,
        apply_fn=_activate_backup_rw,
    ),
    "deploy_radiators": ActionSpec(
        name="deploy_radiators",
        description="Deploy radiator panels and open the thermal loop.",
        human_label="Deploy radiators",
        params={},
        battery_delta_pct=-0.3,
        requires_comms_path=None,
        apply_fn=_deploy_radiators,
    ),
    "reduce_compute_load": ActionSpec(
        name="reduce_compute_load",
        description="Reduce PCU computational load by the given percentage.",
        human_label="Reduce compute load",
        params={"pct": {"type": "number", "range": (10, 60)}},
        battery_delta_pct=+0.6,
        requires_comms_path=None,
        apply_fn=_reduce_compute_load,
    ),
    "bypass_solar_string": ActionSpec(
        name="bypass_solar_string",
        description="Route around a failed solar array string via the surviving strings.",
        human_label="Bypass failed solar string",
        params={},
        battery_delta_pct=-0.2,
        requires_comms_path=None,
        apply_fn=_bypass_solar_string,
    ),
    "hold_and_observe": ActionSpec(
        name="hold_and_observe",
        description="Take no state-changing action; monitor telemetry for the given duration.",
        human_label="Hold and observe",
        params={"seconds": {"type": "number", "range": (5, 300)}},
        battery_delta_pct=+0.05,
        requires_comms_path=None,
        apply_fn=_hold_and_observe,
    ),
}


def action_vocabulary_for_prompt() -> str:
    """A compact vocabulary description Gemma can hold in-context.

    Kept short deliberately — the whole point of tiering by role is that each
    tier gets a tightly-scoped prompt. A tier that has to reason about the
    entire spec of every action is a tier that will ramble.
    """
    lines = []
    for spec in ACTIONS.values():
        params = ""
        if spec.params:
            pieces = []
            for pname, pspec in spec.params.items():
                if "choices" in pspec:
                    pieces.append(f"{pname} ∈ {{{'|'.join(sorted(pspec['choices']))}}}")
                elif "range" in pspec:
                    lo, hi = pspec["range"]
                    pieces.append(f"{pname} ∈ [{lo},{hi}]")
                else:
                    pieces.append(pname)
            params = f"({', '.join(pieces)})"
        lines.append(f"  {spec.name}{params} — {spec.description}")
    return "\n".join(lines)


def apply_plan_step(state: SpacecraftState, action_name: str, params: dict) -> SpacecraftState:
    """Return a new state with the named action applied.

    Never mutates the input state — planning previews and validator dry-runs
    must not disturb the live spacecraft.
    """
    if action_name not in ACTIONS:
        raise ValueError(f"Unknown action: {action_name}")
    new_state = copy_state(state)
    return ACTIONS[action_name].apply_fn(new_state, params)
