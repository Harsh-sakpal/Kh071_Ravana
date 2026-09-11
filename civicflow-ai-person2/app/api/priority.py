"""
CivicFlow AI - Priority Agent API
===================================

Independent REST endpoint for automated ticket priority scoring and SLA determination.
Callable and testable on its own per Section 6.
"""

from fastapi import APIRouter

from app.models.priority import PriorityRequest, PriorityResult
from app.agents.priority_agent import evaluate_priority

router = APIRouter(prefix="/api/v1/ai", tags=["priority"])


@router.post("/priority", response_model=PriorityResult)
def priority_endpoint(request: PriorityRequest) -> PriorityResult:
    return evaluate_priority(request)
