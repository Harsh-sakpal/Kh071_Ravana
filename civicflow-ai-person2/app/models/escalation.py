"""
CivicFlow AI - Escalation Agent Pydantic Models
===================================================

Request and response schemas for the Advanced Escalation Agent (ESCALATION_AGENT),
supporting 5 multi-stage escalation tiers, officer reassignment candidate selection,
financial/compliance penalty scoring, notification previews, and audit traces.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class OfficerCandidate(BaseModel):
    officer_id: str = Field(..., description="Unique officer employee ID.")
    name: str = Field(..., description="Full officer name.")
    active_tickets: int = Field(default=0, ge=0, description="Current number of active tickets assigned.")
    skills: List[str] = Field(default_factory=list, description="Officer specialized skills.")
    is_available: bool = Field(default=True, description="True if officer is currently on active shift.")


class EscalationRequest(BaseModel):
    ticket_id: Optional[str] = Field(default="TCK-DEMO-2026", description="Municipal tracking ticket UUID.")
    complaint_text: Optional[str] = Field(default=None, description="Raw complaint description.")
    department_code: Optional[str] = Field(default="SAN", description="Target department code (e.g. SAN, RDW, WTR, ELE, PRK).")
    priority: str = Field(default="MEDIUM", description="Assigned priority level: LOW, MEDIUM, HIGH, or CRITICAL.")
    sla_status: str = Field(default="WITHIN_SLA", description="SLA status from SLA Agent: WITHIN_SLA, NEAR_BREACH, BREACHED, EXEMPT.")
    sla_tier: str = Field(default="ON_TRACK", description="SLA multi-stage tier: ON_TRACK, PROGRESSING, NEAR_BREACH, CRITICAL_RISK, BREACHED, EXEMPT.")
    sla_ratio: float = Field(default=0.0, ge=0.0, description="Ratio of elapsed SLA hours to target SLA hours.")
    breach_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Evaluated continuous 6-factor breach probability.")
    is_breached: bool = Field(default=False, description="True if ticket has exceeded target SLA duration.")
    is_safety_override: bool = Field(default=False, description="True if continuous DIPEX safety override was triggered.")
    affected_count: Optional[int] = Field(default=0, ge=0, description="Estimated number of affected citizens.")
    current_officer_id: Optional[str] = Field(default="OFF-ASSIGNED-01", description="Currently assigned officer ID.")
    candidate_officers: Optional[List[OfficerCandidate]] = Field(
        default=None,
        description="Optional list of available candidate officers for automated reassignment recommendation.",
    )


class EscalationResult(BaseModel):
    status: str = Field(..., description="SUCCESS or ERROR.")
    ticket_id: Optional[str] = None
    escalation_tier: str = Field(
        ...,
        description="Evaluated escalation level: TIER_0_NOMINAL, TIER_1_SUPERVISOR_ALERT, TIER_2_PRE_BREACH_REASSIGNMENT, TIER_3_DEPARTMENT_HEAD_ESCALATION, TIER_4_EMERGENCY_OVERRIDE.",
    )
    action_required: str = Field(
        ...,
        description="Triggered action: NONE, SUPERVISOR_NOTIFIED, REASSIGNMENT_RECOMMENDED, EXEC_BOARD_ESCALATED, EMERGENCY_DISPATCH_TRIGGERED.",
    )
    target_role: str = Field(
        ...,
        description="Target recipient: ASSIGNED_OFFICER, SUPERVISOR, REASSIGNED_OFFICER, DEPARTMENT_HEAD, EMERGENCY_RESPONSE_UNIT.",
    )
    target_contact: str = Field(..., description="Recipient email, phone, or office dispatch identifier.")
    recommended_officer_id: Optional[str] = Field(default=None, description="Recommended officer ID for reassignment if Tier >= 2.")
    recommended_officer_name: Optional[str] = Field(default=None, description="Recommended officer name for reassignment if Tier >= 2.")
    penalty_score: float = Field(default=0.0, ge=0.0, description="Calculated SLA breach penalty points / compliance impact score.")
    notification_draft: str = Field(..., description="Simulated notification message (SMS/email draft) for target recipient.")
    audit_trace: List[str] = Field(default_factory=list, description="Log of audit events triggered by escalation engine.")
    explanation: str = Field(..., description="Human-readable summary of escalation decision rationale.")
    warnings: List[str] = Field(default_factory=list, description="Warnings or non-fatal alerts encountered.")
