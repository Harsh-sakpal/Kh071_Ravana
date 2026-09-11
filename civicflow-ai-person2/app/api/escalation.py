"""
CivicFlow AI - Escalation Agent API
======================================

REST endpoint for automated ticket escalation, officer reassignment recommendations,
SLA breach penalty scoring, and target notification previews.
"""

from fastapi import APIRouter

from app.agents.escalation_agent import evaluate_escalation
from app.models.escalation import EscalationRequest, EscalationResult

router = APIRouter(prefix="/api/v1/ai", tags=["escalation"])


@router.post("/escalate", response_model=EscalationResult)
def escalate_endpoint(request: EscalationRequest) -> EscalationResult:
    return evaluate_escalation(request)
