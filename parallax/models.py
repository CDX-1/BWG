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
    evidence_needed: list[str]


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
