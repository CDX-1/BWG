from typing import Literal
from pydantic import BaseModel, Field


class EvidenceAssessment(BaseModel):
    evidence_id: str
    priority: int = Field(ge=1, le=5)
    reason: str
    supports_hypotheses: list[str]
    consequence_if_lost: str


class Hypothesis(BaseModel):
    name: str
    confidence: Literal["low", "medium", "high"]
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    evidence_needed: list[str] = []
    # A citation is a section tag like "[INCIDENT-RESPONSE §2.1]". G3 flags
    # any hypothesis that doesn't cite at least one procedure section.
    citations: list[str] = []


class SourceReference(BaseModel):
    source_id: str
    relevance: str


class ParallaxAnalysis(BaseModel):
    event_summary: str
    resolution_status: Literal[
        "known_event", "likely_fault", "likely_scientific_event", "unresolved"
    ]
    hypotheses: list[Hypothesis]
    preservation_priorities: list[EvidenceAssessment]
    recommended_follow_up: str
    compression_warning: str
    uncertainty_statement: str
    source_references: list[SourceReference] = []


class FDIRActionModel(BaseModel):
    timestamp: str
    fault_id: str
    phase: str
    message: str
    outcome: str


class FDIRReportModel(BaseModel):
    triggered: bool
    active_faults: list[str]
    actions: list[FDIRActionModel]
    mission_safe: bool
    summary: str


class PredictedFailure(BaseModel):
    subsystem: str
    failure_mode: str
    estimated_time_to_failure: str
    probability: Literal["low", "medium", "high"]
    early_warning_signs: list[str]


class FixOption(BaseModel):
    name: str
    description: str
    success_pct: int = Field(default=70, ge=0, le=100)
    risk: Literal["low", "medium", "high"] = "medium"
    autonomous: bool = True


class GemmaPredictiveAnalysis(BaseModel):
    """Legacy per-fault predictive output — retained for the fallback cache path."""
    current_assessment: str
    system_stability: Literal["stable", "degraded", "critical", "unknown"]
    earth_delay_min: int = 38
    basic_fix_success_pct: int = Field(default=70, ge=0, le=100)
    cascade_failure_pct: int = Field(default=25, ge=0, le=100)
    predicted_failures: list[PredictedFailure]
    cascading_risks: list[str]
    alternative_fixes: list[FixOption] = []
    chosen_fix: str = "Basic FDIR Recovery"
    execute_immediately: bool = True
    execute_reason: str = ""
    recommended_actions: list[str]
    earth_report: str
    confidence: Literal["low", "medium", "high"]


# ── Five-tier stack outputs ─────────────────────────────────────────────────

class SentinelVerdict(BaseModel):
    """G0 — one-word situational assessment on the live telemetry window."""
    status: Literal["nominal", "watch", "anomalous"]
    what_looks_odd: str = ""


class DiagnosticianHypothesis(BaseModel):
    """G1 — a single competing explanation, evidence-supported, cited."""
    name: str
    confidence: Literal["low", "medium", "high"]
    supporting_evidence: list[str] = []
    contradicting_evidence: list[str] = []
    citations: list[str] = []


class Diagnosis(BaseModel):
    """G1 output — 2–4 competing hypotheses, uncertainty preserved."""
    summary: str
    hypotheses: list[DiagnosticianHypothesis]
    # Whether G1 could ground every hypothesis in cited procedure text.
    all_hypotheses_cited: bool = True


class PlanStep(BaseModel):
    """G2 — one action in the recovery plan."""
    action: str
    params: dict = {}
    rationale: str = ""
    citations: list[str] = []


class RecoveryPlan(BaseModel):
    """G2 output — sequenced plan over the whitelisted vocabulary."""
    summary: str
    steps: list[PlanStep]
    estimated_battery_cost_pct: float = 0.0
    addresses_hypotheses: list[str] = []


class AdjudicatorVerdict(BaseModel):
    """G3 output — cold, temp-0 red-team of G2's plan."""
    approved: bool
    reason: str
    # Which specific step G3 objected to, if any (1-indexed).
    objection_step: int | None = None
    # Optional recommendations G3 would want to see in a re-plan.
    revision_notes: list[str] = []


class EvidenceScore(BaseModel):
    """G4 — one record scored for downlink survival value."""
    evidence_id: str
    score: int = Field(ge=0, le=100)
    reason: str
    supports_hypotheses: list[str] = []


class ArchivistPacking(BaseModel):
    """G4 output — value model for the knapsack capsule."""
    scores: list[EvidenceScore]
    strategy: str = ""


class TierTrace(BaseModel):
    """One entry in the pipeline trace shown in the UI."""
    tier: Literal["G0", "G1", "G2", "G3", "G4"]
    label: str
    ok: bool
    is_live: bool
    latency_s: float
    tokens_out: int = 0
    note: str = ""
    error: str | None = None
