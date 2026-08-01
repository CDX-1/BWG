"""Deterministic gates that stand between G2's plan and the spacecraft.

Every plan step passes three gates before any state changes:
  1. schema     — the step names an action in the vocabulary, with valid params
  2. whitelist  — the referenced action exists and is currently allowed
  3. envelope   — the projected end-state does not violate a hard safety rule

A rejected plan is not a failure to display; it is the demo. The UI shows the
failing gate and the reason, and asks G2 to re-plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from parallax.action_vocabulary import ACTIONS, apply_plan_step
from parallax.spacecraft import SpacecraftState, copy_state


# ── Hard safety envelope (never violated no matter what G2 proposes) ────────

# The battery may not project below this floor by the end of the plan.
BATTERY_FLOOR_PCT = 40.0

# The bus voltage may not project below this floor by any step.
BUS_VOLTAGE_FLOOR_V = 22.5

# If the HGA is already down, an antenna step that leaves *both* paths broken
# is rejected — the demo must show this specific class of veto.
COMMS_PATHS = ("hga", "lga_fallback")


@dataclass
class GateResult:
    step_index: int
    action_name: str
    gate: str          # "schema" | "whitelist" | "envelope"
    passed: bool
    reason: str = ""


@dataclass
class ValidationReport:
    approved: bool
    rejected_step_index: Optional[int]
    rejected_gate: Optional[str]
    rejected_reason: Optional[str]
    per_step: list[GateResult] = field(default_factory=list)
    projected_state: Optional[SpacecraftState] = None

    @property
    def failing_gate(self) -> Optional[str]:
        return self.rejected_gate

    @property
    def summary_line(self) -> str:
        if self.approved:
            n = len(self.per_step) // 3   # three gates per step
            return f"APPROVED · {n} step(s) cleared all gates"
        return (f"REJECTED · step {self.rejected_step_index + 1} "
                f"failed {self.rejected_gate} gate · {self.rejected_reason}")


def _schema_check(step_index: int, action_name: str, params: dict) -> GateResult:
    spec = ACTIONS.get(action_name)
    if spec is None:
        return GateResult(step_index, action_name, "schema", False,
                          f"'{action_name}' is not in the action vocabulary")

    required = set(spec.params)
    supplied = set(params or {})
    missing = required - supplied
    extra = supplied - required
    if missing:
        return GateResult(step_index, action_name, "schema", False,
                          f"missing parameter(s): {', '.join(sorted(missing))}")
    if extra:
        return GateResult(step_index, action_name, "schema", False,
                          f"unknown parameter(s): {', '.join(sorted(extra))}")

    for pname, pspec in spec.params.items():
        value = params.get(pname)
        if pspec.get("type") == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return GateResult(step_index, action_name, "schema", False,
                                  f"{pname} must be numeric, got {type(value).__name__}")
            if "range" in pspec:
                lo, hi = pspec["range"]
                if not (lo <= value <= hi):
                    return GateResult(step_index, action_name, "schema", False,
                                      f"{pname}={value} outside [{lo},{hi}]")
        elif pspec.get("type") == "string":
            if not isinstance(value, str):
                return GateResult(step_index, action_name, "schema", False,
                                  f"{pname} must be a string, got {type(value).__name__}")
            choices = pspec.get("choices")
            if choices and value not in choices:
                return GateResult(step_index, action_name, "schema", False,
                                  f"{pname}={value!r} not in {sorted(choices)}")
    return GateResult(step_index, action_name, "schema", True)


def _whitelist_check(step_index: int, action_name: str) -> GateResult:
    # The action vocabulary itself is the whitelist. In a real spacecraft this
    # gate would additionally consult a per-mission-mode capability list.
    if action_name not in ACTIONS:
        return GateResult(step_index, action_name, "whitelist", False,
                          "action is not on the vehicle's approved command list")
    return GateResult(step_index, action_name, "whitelist", True)


def _envelope_check(
    step_index: int,
    action_name: str,
    before_state: SpacecraftState,
    projected: SpacecraftState,
) -> GateResult:
    if projected.battery_soc_pct < BATTERY_FLOOR_PCT:
        return GateResult(step_index, action_name, "envelope", False,
                          f"projected battery {projected.battery_soc_pct:.1f}% would fall below "
                          f"the {BATTERY_FLOOR_PCT:.0f}% mission floor")

    if projected.bus_voltage_v < BUS_VOLTAGE_FLOOR_V:
        return GateResult(step_index, action_name, "envelope", False,
                          f"projected bus voltage {projected.bus_voltage_v:.1f} V would fall below "
                          f"the {BUS_VOLTAGE_FLOOR_V:.1f} V floor")

    # An antenna switch is only unsafe if it would leave the spacecraft with
    # no working comms path at all. HGA is broken if the Communications
    # subsystem is failed; LGA is our permanent fallback.
    if action_name == "switch_antenna":
        target = getattr(projected, "antenna_mode", "hga")
        hga_health = before_state.subsystem_health.get("Communications", "nominal")
        if target == "hga" and hga_health == "failed":
            return GateResult(step_index, action_name, "envelope", False,
                              "cannot return to HGA while Communications subsystem is failed")

    return GateResult(step_index, action_name, "envelope", True)


def validate_plan(
    steps: list[dict],
    current_state: SpacecraftState,
) -> ValidationReport:
    """Run the three gates over each step in order.

    Steps are checked cumulatively: the envelope gate runs against the
    projected state *after* every previous step has been applied. That is
    exactly what makes a plan like "shed 40% then shed 40% again" fail —
    the second step's envelope check sees the depleted battery from the first.
    """
    per_step: list[GateResult] = []
    projected = copy_state(current_state)

    for idx, step in enumerate(steps):
        action_name = step.get("action", "")
        params = step.get("params", {}) or {}

        schema = _schema_check(idx, action_name, params)
        per_step.append(schema)
        if not schema.passed:
            return _reject(idx, "schema", schema.reason, per_step, projected)

        whitelist = _whitelist_check(idx, action_name)
        per_step.append(whitelist)
        if not whitelist.passed:
            return _reject(idx, "whitelist", whitelist.reason, per_step, projected)

        try:
            new_projected = apply_plan_step(projected, action_name, params)
        except Exception as exc:
            reason = f"apply_fn raised {type(exc).__name__}: {exc}"
            envelope = GateResult(idx, action_name, "envelope", False, reason)
            per_step.append(envelope)
            return _reject(idx, "envelope", reason, per_step, projected)

        envelope = _envelope_check(idx, action_name, projected, new_projected)
        per_step.append(envelope)
        if not envelope.passed:
            return _reject(idx, "envelope", envelope.reason, per_step, projected)

        projected = new_projected

    return ValidationReport(
        approved=True,
        rejected_step_index=None,
        rejected_gate=None,
        rejected_reason=None,
        per_step=per_step,
        projected_state=projected,
    )


def _reject(step_index: int, gate: str, reason: str,
            per_step: list[GateResult], projected: SpacecraftState) -> ValidationReport:
    return ValidationReport(
        approved=False,
        rejected_step_index=step_index,
        rejected_gate=gate,
        rejected_reason=reason,
        per_step=per_step,
        projected_state=projected,
    )


def apply_approved_plan(
    steps: list[dict],
    current_state: SpacecraftState,
) -> SpacecraftState:
    """Commit every step of an already-validated plan to a fresh state copy."""
    projected = copy_state(current_state)
    for step in steps:
        projected = apply_plan_step(projected, step["action"], step.get("params", {}) or {})
    return projected
