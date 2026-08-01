"""Scenario harness — proves the five-tier stack beats bare FDIR.

Four measurable metrics, all listed in the plan:
  1. Conflicting-recovery detection   — compound faults where FAULT_RECOVERY
     scripts contradict each other. Bare FDIR concatenates them; the stack's
     G3 catches the conflict.
  2. Unsafe plans blocked             — how many G2 plans were rejected by
     the validator or G3.
  3. Plan quality delta               — before/after health-score delta for
     validated plans versus bare FDIR outcomes.
  4. Time to first actionable decision — wall clock, versus the 38-minute
     Earth round trip that makes onboard autonomy necessary at all.

The harness is deterministic in scenario choice (fixed list) and reports
failures honestly. A stack that shows "G3 vetoed 2 of 9 plans" is far more
credible than one claiming a clean sweep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from parallax.fdir import run_fdir
from parallax.featurize import featurize_spacecraft_state
from parallax.spacecraft import copy_state, inject_fault, rebuild_state, SpacecraftState
from parallax.tiers import run_stack


# A modest but pointed set: single faults, and compound faults whose scripted
# recoveries fight for the same power/pointing budget.
SCENARIOS: list[dict] = [
    {"id": "s01_solar",  "faults": ["solar_string_loss"], "label": "Solar string failure"},
    {"id": "s02_thermal","faults": ["thermal_runaway"],    "label": "Thermal runaway"},
    {"id": "s03_pcu",    "faults": ["pcu_fault"],           "label": "PCU electronics fault"},
    {"id": "s04_rw",     "faults": ["reaction_wheel_fault"],"label": "Reaction wheel"},
    {"id": "s05_spec",   "faults": ["spectrometer_fault"],  "label": "Spectrometer corruption"},
    {"id": "s06_comms",  "faults": ["comms_dropout"],       "label": "Comms dropout"},
    # ── Compound / conflict scenarios ──────────────────────────────────────
    {"id": "s07_therm_adcs", "faults": ["thermal_runaway", "reaction_wheel_fault"],
     "label": "Thermal + RW (compete for load-shed vs. slew)",
     "conflict": True},
    {"id": "s08_pcu_comms",  "faults": ["pcu_fault", "comms_dropout"],
     "label": "PCU + Comms (backup PCU brownout risk vs. LGA acquisition)",
     "conflict": True},
    {"id": "s09_solar_therm","faults": ["solar_string_loss", "thermal_runaway"],
     "label": "Solar + Thermal (recovery draws more heat)",
     "conflict": True},
]

# Simple health scoring: nominal=3, recovering=2, degraded=1, failed=0.
HEALTH_SCORE = {"nominal": 3, "recovering": 2, "degraded": 1, "failed": 0}


@dataclass
class ScenarioOutcome:
    id: str
    label: str
    fault_count: int
    is_conflict: bool

    fdir_only_score: int
    stack_score: int
    stack_plan_iterations: int
    validator_rejected: bool
    adjudicator_vetoed: bool
    wall_time_s: float
    plan_step_count: int

    fdir_only_summary: str = ""
    stack_summary: str = ""


@dataclass
class BenchmarkReport:
    outcomes: list[ScenarioOutcome] = field(default_factory=list)
    total_wall_s: float = 0.0

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def conflicts_caught_by_stack(self) -> int:
        # A conflict is "caught" when the stack's plan iterations > 1 (a veto
        # forced a re-plan) OR the stack's post-plan score beats FDIR-only's.
        return sum(
            1 for o in self.outcomes
            if o.is_conflict and (o.stack_plan_iterations > 1 or o.stack_score > o.fdir_only_score)
        )

    @property
    def total_conflicts(self) -> int:
        return sum(1 for o in self.outcomes if o.is_conflict)

    @property
    def unsafe_plans_blocked(self) -> int:
        return sum(1 for o in self.outcomes if o.validator_rejected or o.adjudicator_vetoed)

    @property
    def stack_wins(self) -> int:
        return sum(1 for o in self.outcomes if o.stack_score > o.fdir_only_score)

    @property
    def mean_wall_time_s(self) -> float:
        return self.total_wall_s / max(1, self.n)


def _score_state(state) -> int:
    return sum(HEALTH_SCORE.get(v, 0) for v in state.subsystem_health.values())


def run_one(scenario: dict, include_sentinel: bool = False) -> ScenarioOutcome:
    """Run one scenario through FDIR-only and through the stack.

    FDIR-only uses the existing spacecraft.rebuild_state + fdir.run_fdir path.
    The stack path runs after FDIR: its plan modifies the FDIR post-recovery
    state, and the score is measured on the resulting state.
    """
    fault_ids: list[str] = scenario["faults"]

    # ── FDIR-only ─────────────────────────────────────────────────────────
    fdir_state = rebuild_state(fault_ids)
    fdir_report = run_fdir(fdir_state)
    fdir_only_score = _score_state(fdir_report.post_recovery_state)

    # ── Stack ─────────────────────────────────────────────────────────────
    stack_state = rebuild_state(fault_ids)
    _ = run_fdir(stack_state)                    # FDIR runs first, feeding the stack

    featurised = featurize_spacecraft_state(stack_state, samples=None)
    evidence = _synthetic_evidence(fault_ids)

    t0 = time.perf_counter()
    stack_result = run_stack(
        featurised_state=featurised,
        fdir_summary=fdir_report.summary,
        active_faults=fault_ids,
        available_evidence=evidence,
        current_state=stack_state,
        include_sentinel=include_sentinel,
        max_replans=1,
    )
    wall_s = time.perf_counter() - t0

    # If the plan is validated (or a re-plan validated), apply it and re-score.
    if stack_result.validation and stack_result.validation.approved and stack_result.verdict and stack_result.verdict.approved:
        final_state = stack_result.projected_state
    else:
        final_state = fdir_report.post_recovery_state
    stack_score = _score_state(final_state)

    return ScenarioOutcome(
        id=scenario["id"],
        label=scenario["label"],
        fault_count=len(fault_ids),
        is_conflict=bool(scenario.get("conflict")),
        fdir_only_score=fdir_only_score,
        stack_score=stack_score,
        stack_plan_iterations=stack_result.plan_iterations,
        validator_rejected=bool(stack_result.validation and not stack_result.validation.approved),
        adjudicator_vetoed=bool(stack_result.verdict and not stack_result.verdict.approved),
        wall_time_s=wall_s,
        plan_step_count=len(stack_result.plan.steps) if stack_result.plan else 0,
        fdir_only_summary=fdir_report.summary,
        stack_summary=(stack_result.plan.summary if stack_result.plan else ""),
    )


def run_all(scenarios: list[dict] | None = None,
            include_sentinel: bool = False) -> BenchmarkReport:
    scenarios = scenarios or SCENARIOS
    outcomes: list[ScenarioOutcome] = []
    total = time.perf_counter()
    for scenario in scenarios:
        outcomes.append(run_one(scenario, include_sentinel=include_sentinel))
    return BenchmarkReport(outcomes=outcomes, total_wall_s=time.perf_counter() - total)


# ── Synthetic evidence packages for the archivist / capsule side ────────────

_EVIDENCE_BY_SUBSYSTEM: dict[str, list[dict]] = {
    "Power":         [
        {"id": "power_window_60s", "size_bytes": 4200, "description": "Power bus + solar window 60 s"},
        {"id": "battery_soc_full", "size_bytes": 2100, "description": "Battery state-of-charge full history"},
    ],
    "Thermal":       [
        {"id": "thermal_window_60s", "size_bytes": 3800, "description": "PCU/bus/instrument temperature 60 s"},
        {"id": "radiator_log",       "size_bytes": 1500, "description": "Radiator deploy/thermal control log"},
    ],
    "ADCS":          [
        {"id": "rw_torque_history", "size_bytes": 4500, "description": "Reaction wheel torque history"},
        {"id": "attitude_error_ts", "size_bytes": 3200, "description": "Attitude error time series"},
    ],
    "Communications":[
        {"id": "link_margin_ts", "size_bytes": 2500, "description": "Link margin over the fault window"},
        {"id": "antenna_state_log", "size_bytes": 1200, "description": "HGA/LGA switching log"},
    ],
    "Science":       [
        {"id": "raw_spectrum",     "size_bytes": 14000, "description": "Raw spectrometer frames"},
        {"id": "camera_thumbnail", "size_bytes": 8000,  "description": "Context camera thumbnail"},
        {"id": "radiation",        "size_bytes": 3500,  "description": "Radiation counter time series"},
    ],
}

# Every scenario also ships packet-integrity — small, irreplaceable.
_ALWAYS_INCLUDE = [
    {"id": "packet_checksums", "size_bytes": 1200,
     "description": "CRC + sequence number log for the fault window"},
]

_FAULT_SUBSYSTEM = {
    "solar_string_loss": "Power",
    "pcu_fault": "Power",
    "reaction_wheel_fault": "ADCS",
    "spectrometer_fault": "Science",
    "comms_dropout": "Communications",
    "thermal_runaway": "Thermal",
}


def _synthetic_evidence(fault_ids: list[str]) -> list[dict]:
    """Build a plausible evidence list for a given fault combination."""
    evidence = list(_ALWAYS_INCLUDE)
    seen = {e["id"] for e in evidence}
    for fid in fault_ids:
        subsystem = _FAULT_SUBSYSTEM.get(fid)
        if not subsystem:
            continue
        for item in _EVIDENCE_BY_SUBSYSTEM.get(subsystem, []):
            if item["id"] not in seen:
                evidence.append(item)
                seen.add(item["id"])
    # Include raw_spectrum for compound scenarios even when Science isn't a
    # named fault — a downlink capsule always wants the raw high-value data.
    if not any(e["id"] == "raw_spectrum" for e in evidence):
        for item in _EVIDENCE_BY_SUBSYSTEM["Science"]:
            if item["id"] not in seen:
                evidence.append(item)
                seen.add(item["id"])
    return evidence
