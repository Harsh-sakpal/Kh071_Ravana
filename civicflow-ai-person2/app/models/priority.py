"""
CivicFlow AI - Priority Agent Pydantic Models
===============================================

Defines input/output schemas for the upgraded Priority Agent API, supporting
DIPEX-style continuous scoring (0.0 - 1.0), factor breakdowns, safety overrides,
and optional SLA/hotspot/risk data integrations.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SLAData(BaseModel):
    sla_duration_hours: Optional[int] = Field(default=None, description="Configured SLA duration in hours.")
    remaining_sla_hours: Optional[float] = Field(default=None, description="Remaining hours before SLA breach.")
    is_near_breach: Optional[bool] = Field(default=False, description="Flag indicating proximity to SLA breach.")
    is_breached: Optional[bool] = Field(default=False, description="Flag indicating SLA is already breached.")


class HotspotData(BaseModel):
    is_hotspot: Optional[bool] = Field(default=False, description="True if location is a detected hotspot.")
    complaint_density: Optional[float] = Field(default=None, ge=0.0, description="Local complaint density score.")
    cluster_count: Optional[int] = Field(default=None, ge=0, description="Count of similar complaints in cluster.")
    trend: Optional[str] = Field(default=None, description="Trend indicator (e.g. INCREASING, STABLE).")


class RiskData(BaseModel):
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Aggregated risk engine score.")
    risk_level: Optional[str] = Field(default=None, description="Risk level (LOW, MEDIUM, HIGH, CRITICAL).")
    safety_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Public safety risk component.")
    health_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Public health risk component.")
    infrastructure_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Infrastructure risk component.")


class PriorityRequest(BaseModel):
    complaint_id: Optional[str] = Field(
        default=None,
        description="Optional upstream complaint UUID/tracking ID.",
    )
    complaint_text: str = Field(
        ...,
        min_length=3,
        description="The raw text description of the civic issue.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Optional pre-classified domain (e.g. WATER_SUPPLY).",
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="Optional pre-classified subcategory (e.g. OPEN_MANHOLE).",
    )
    category_code: Optional[str] = Field(
        default=None,
        description="Optional database category code (e.g. SD-002, WTR-LEAK).",
    )
    location_context: Optional[str] = Field(
        default=None,
        description="Optional location notes (e.g. 'near school', 'busy intersection').",
    )
    affected_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional estimated number of citizens affected.",
    )
    duration_days: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional duration the problem has existed in days.",
    )
    recurrence_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional count of past repeated complaints in the same location/category.",
    )
    sla_info: Optional[SLAData] = Field(
        default=None,
        description="Optional SLA tracking context.",
    )
    hotspot_info: Optional[HotspotData] = Field(
        default=None,
        description="Optional Hotspot Engine analytical context.",
    )
    risk_info: Optional[RiskData] = Field(
        default=None,
        description="Optional Risk Engine analytical context.",
    )


class PriorityFactors(BaseModel):
    severity: float = Field(..., ge=0.0, le=1.0, description="Complaint severity factor score.")
    public_impact: float = Field(..., ge=0.0, le=1.0, description="Public impact factor score.")
    urgency: float = Field(..., ge=0.0, le=1.0, description="Urgency signal factor score.")
    infrastructure_criticality: float = Field(..., ge=0.0, le=1.0, description="Critical infrastructure factor score.")
    affected_population: float = Field(..., ge=0.0, le=1.0, description="Affected population scale factor score.")
    recurrence: float = Field(..., ge=0.0, le=1.0, description="Recurrence frequency factor score.")
    duration: float = Field(..., ge=0.0, le=1.0, description="Complaint duration factor score.")
    sla_risk: float = Field(..., ge=0.0, le=1.0, description="SLA breach risk factor score.")
    hotspot: float = Field(..., ge=0.0, le=1.0, description="Hotspot density factor score.")
    risk: float = Field(..., ge=0.0, le=1.0, description="Risk engine factor score.")


class PriorityResult(BaseModel):
    status: str = Field(
        ...,
        description="SUCCESS or LOW_CONFIDENCE.",
    )
    complaint_id: Optional[str] = None
    priority: str = Field(
        ...,
        description="Assigned ticket priority: LOW, MEDIUM, HIGH, or CRITICAL.",
    )
    priority_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous DIPEX-style priority score from 0.0 to 1.0.",
    )
    sla_target_hours: int = Field(
        ...,
        description="Target resolution SLA in hours (6 for CRITICAL, 24 for HIGH, 48 for MEDIUM, 120 for LOW).",
    )
    factors: PriorityFactors = Field(
        ...,
        description="Normalized feature score breakdown across 10 DIPEX feature groups.",
    )
    override_applied: bool = Field(
        default=False,
        description="True if a deterministic safety override forced CRITICAL priority.",
    )
    override_reason: Optional[str] = Field(
        default=None,
        description="Reason for deterministic safety override, if applied.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the priority assessment based on input clarity & signal consistency.",
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of priority calculation and contributing factors.",
    )
    reason: str = Field(
        ...,
        description="Alias for explanation for backward compatibility.",
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="List of detected high-risk signals.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings or non-fatal alerts encountered during processing.",
    )
