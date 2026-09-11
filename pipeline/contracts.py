"""
CivicFlow AI — Unified Pipeline Contracts
============================================

Pydantic models for the end-to-end orchestrator that bridges Person 1
(AI/NLP classification) with Person 2 (Routing, Priority, SLA, Escalation).

These contracts define the single input and unified output for the entire
8-agent pipeline.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PipelineInput(BaseModel):
    """Single entry point for the entire CivicFlow pipeline."""

    complaint_text: str = Field(
        ..., min_length=3, description="Raw citizen complaint text."
    )
    complaint_id: Optional[str] = Field(
        default=None, description="Optional tracking ID for traceability."
    )
    location_context: Optional[str] = Field(
        default=None, description="Optional location notes (e.g. 'near school')."
    )
    affected_count: Optional[int] = Field(
        default=None, ge=0, description="Estimated number of citizens affected."
    )
    duration_days: Optional[float] = Field(
        default=None, ge=0.0, description="How long the issue has persisted (days)."
    )
    recurrence_count: Optional[int] = Field(
        default=None, ge=0, description="Number of past complaints for this issue."
    )
    debug: bool = Field(
        default=False, description="If True, include trace/debug information."
    )


class ClassificationResult(BaseModel):
    """Person 1 output — AI/NLP classification."""

    status: str = "UNKNOWN"
    domain_code: Optional[str] = None
    domain_name: Optional[str] = None
    category_code: Optional[str] = None
    category_name: Optional[str] = None
    department_code: Optional[str] = None
    mapping_type: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    normalized_domain: Optional[str] = Field(
        default=None,
        description="Person 2 canonical domain after alias normalization.",
    )


class RoutingResult(BaseModel):
    """Person 2 Routing Agent output."""

    status: str = "PENDING"
    domain: Optional[str] = None
    subcategory: Optional[str] = None
    category_code: Optional[str] = None
    department_code: Optional[str] = None
    confidence: float = 0.0
    reason: Optional[str] = None
    category_mapping_available: bool = False
    warnings: List[str] = Field(default_factory=list)


class PriorityResult(BaseModel):
    """Person 2 Priority Agent output."""

    status: str = "PENDING"
    priority: Optional[str] = None
    priority_score: float = 0.0
    sla_target_hours: Optional[int] = None
    factors: Optional[Dict[str, float]] = None
    override_applied: bool = False
    override_reason: Optional[str] = None
    confidence: float = 0.0
    explanation: Optional[str] = None
    risk_factors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SLAResult(BaseModel):
    """Person 2 SLA Monitoring Agent output."""

    status: str = "PENDING"
    sla_status: Optional[str] = None
    sla_tier: Optional[str] = None
    elapsed_hours: float = 0.0
    remaining_hours: float = 0.0
    sla_target_hours: Optional[int] = None
    sla_ratio: float = 0.0
    is_breached: bool = False
    breach_probability: float = 0.0
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class EscalationResult(BaseModel):
    """Person 2 Escalation Agent output."""

    status: str = "PENDING"
    escalation_tier: Optional[str] = None
    action_required: Optional[str] = None
    target_role: Optional[str] = None
    target_contact: Optional[str] = None
    recommended_officer_id: Optional[str] = None
    recommended_officer_name: Optional[str] = None
    penalty_score: float = 0.0
    notification_draft: Optional[str] = None
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class PipelineOutput(BaseModel):
    """Unified output of the entire 8-agent pipeline."""

    # Meta
    pipeline_status: str = Field(
        default="SUCCESS",
        description="Overall pipeline status: SUCCESS, PARTIAL, ERROR.",
    )
    complaint_id: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pipeline_version: str = "1.0.0"

    # Stage outputs
    classification: ClassificationResult = Field(
        default_factory=ClassificationResult
    )
    routing: RoutingResult = Field(default_factory=RoutingResult)
    priority: PriorityResult = Field(default_factory=PriorityResult)
    sla: SLAResult = Field(default_factory=SLAResult)
    escalation: EscalationResult = Field(default_factory=EscalationResult)

    # Pipeline-level
    agent_chain: List[str] = Field(
        default_factory=list,
        description="Ordered list of agents that executed successfully.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Errors from any failed stages.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Aggregated warnings from all stages.",
    )
    debug_trace: Optional[Dict[str, Any]] = Field(
        default=None, description="Full trace if debug=True."
    )
