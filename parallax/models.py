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
        "known_event",
        "likely_fault",
        "likely_scientific_event",
        "unresolved"
    ]
    hypotheses: list[Hypothesis]
    preservation_priorities: list[EvidenceAssessment]
    recommended_follow_up: str
    compression_warning: str
    uncertainty_statement: str
    source_references: list[SourceReference] = []
