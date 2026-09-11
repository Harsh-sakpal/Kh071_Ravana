"""
CivicFlow AI - SLA Monitoring Agent Pydantic Models
=====================================================

Defines request and response schemas for the Advanced SLA Monitoring Agent,
supporting category SLA rules, location adjustments, 6-factor breach probability
modeling, multi-tier risk statuses, and factor breakdowns.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SLAFactors(BaseModel):
    time_elapsed: float = Field(..., ge=0.0, le=1.0, description="Time elapsed factor score.")
    officer_workload: float = Field(..., ge=0.0, le=1.0, description="Officer workload factor score.")
    department_queue: float = Field(..., ge=0.0, le=1.0, description="Department queue congestion factor score.")
    complexity: float = Field(..., ge=0.0, le=1.0, description="Subcategory work complexity factor score.")
    location_urgency: float = Field(..., ge=0.0, le=1.0, description="Location urgency discount factor score.")
    historical_breach: float = Field(..., ge=0.0, le=1.0, description="Historical breach rate factor score.")


class SLAMonitorRequest(BaseModel):
    ticket_id: Optional[str] = Field(
        default=None,
        description="Optional tracking ticket UUID.",
    )
    complaint_id: Optional[str] = Field(
        default=None,
        description="Optional upstream complaint UUID.",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Creation timestamp of the ticket or complaint. Defaults to current UTC time if not provided.",
    )
    priority: str = Field(
        default="MEDIUM",
        description="Assigned ticket priority: LOW, MEDIUM, HIGH, or CRITICAL.",
    )
    sla_target_hours: Optional[int] = Field(
        default=None,
        ge=1,
        description="Configured target SLA in hours. If omitted, resolved from category rules or priority.",
    )
    category_code: Optional[str] = Field(
        default=None,
        description="Optional database category code (e.g. WTR-LEAK, SWM-001) for custom SLA duration lookup.",
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="Optional subcategory for work complexity factor scoring.",
    )
    status: str = Field(
        default="ASSIGNED",
        description="Current ticket status: OPEN, ASSIGNED, IN_PROGRESS, PENDING_APPROVAL, RESOLVED, CLOSED, REOPENED.",
    )
    officer_workload_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional number of active tickets assigned to the officer.",
    )
    department_queue_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional count of unassigned active tickets in department queue.",
    )
    is_sensitive_location: Optional[bool] = Field(
        default=False,
        description="True if issue is near a school, hospital, or sensitive area (applies 25% SLA target reduction).",
    )
    historical_breach_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional historical category/department breach rate (0.0 to 1.0).",
    )
    current_time: Optional[datetime] = Field(
        default=None,
        description="Optional evaluation timestamp override for deterministic testing.",
    )


class SLAMonitorResult(BaseModel):
    status: str = Field(
        ...,
        description="SUCCESS or ERROR.",
    )
    ticket_id: Optional[str] = None
    sla_status: str = Field(
        ...,
        description="Evaluated SLA status: WITHIN_SLA, NEAR_BREACH, BREACHED, or EXEMPT.",
    )
    sla_tier: str = Field(
        ...,
        description="Multi-stage alert tier: ON_TRACK, PROGRESSING, NEAR_BREACH, CRITICAL_RISK, BREACHED, or EXEMPT.",
    )
    elapsed_hours: float = Field(
        ...,
        ge=0.0,
        description="Number of hours elapsed since ticket creation.",
    )
    remaining_hours: float = Field(
        ...,
        ge=0.0,
        description="Number of hours remaining before SLA breach.",
    )
    sla_target_hours: int = Field(
        ...,
        ge=1,
        description="Configured SLA target resolution hours.",
    )
    sla_ratio: float = Field(
        ...,
        ge=0.0,
        description="Ratio of elapsed time to SLA target hours.",
    )
    is_warning_triggered: bool = Field(
        default=False,
        description="True if elapsed time >= 75% of target SLA and ticket is active.",
    )
    is_breached: bool = Field(
        default=False,
        description="True if elapsed time >= target SLA and ticket is active.",
    )
    breach_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calculated 6-factor continuous breach probability from 0.0 to 1.0.",
    )
    sla_factors: SLAFactors = Field(
        ...,
        description="Normalized score breakdown across 6 SLA risk factor groups.",
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of SLA evaluation, remaining hours, and breach risk breakdown.",
    )
    recommendation_action: str = Field(
        ...,
        description="Operational recommendation action (MAINTAIN_NORMAL_MONITORING, MONITOR_PROGRESS, TRIGGER_NEAR_BREACH_WARNING, PREPARE_PRE_BREACH_ESCALATION, IMMEDIATE_ESCALATION_REQUIRED).",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings or non-fatal alerts encountered during processing.",
    )
