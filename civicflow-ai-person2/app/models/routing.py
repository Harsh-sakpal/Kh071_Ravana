"""
CivicFlow AI - Routing Agent Contracts
========================================

Stable request/response schema for the Routing Agent. Other Person-2
components (Priority, Risk, Orchestrator) will consume RoutingResult, so
this contract should be treated as locked once tests pass (Section 3:
"LOCK CONTRACT" step of the iteration workflow).
"""

from typing import Optional
from pydantic import BaseModel, Field


class RoutingRequest(BaseModel):
    """Minimal input the Routing Agent needs. No unnecessary PII (Section 15)."""

    complaint_id: Optional[str] = Field(
        default=None, description="Optional complaint identifier for traceability/logging."
    )
    complaint_text: str = Field(
        ..., min_length=3, description="Raw or lightly-cleaned complaint description text."
    )
    existing_domain: Optional[str] = Field(
        default=None,
        description="If the Classification Agent already produced a domain, pass it through "
        "so Routing does not duplicate that work.",
    )
    existing_subcategory: Optional[str] = Field(
        default=None,
        description="If the Classification Agent already produced a subcategory.",
    )


class RoutingResult(BaseModel):
    """Structured routing decision. `status` allows partial/failure states so the
    Orchestrator can handle this result without fabricating missing data
    (Section 14)."""

    status: str = Field(..., description="SUCCESS | LOW_CONFIDENCE | UNRESOLVED")
    complaint_id: Optional[str] = None

    domain: Optional[str] = None
    subcategory: Optional[str] = None

    category_code: Optional[str] = Field(
        default=None,
        description="Database categories.code, if a confirmed mapping exists. "
        "Never fabricated -- null means no operational mapping yet.",
    )
    department_code: Optional[str] = Field(
        default=None, description="Database departments.code this complaint should route to."
    )

    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Human-readable explanation of the routing decision.")

    category_mapping_available: bool = Field(
        default=False,
        description="True only if category_code was resolved from a confirmed DB mapping.",
    )
    warnings: list[str] = Field(default_factory=list)
