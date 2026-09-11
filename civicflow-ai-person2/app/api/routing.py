"""
CivicFlow AI - Routing Agent API
==================================

Independent endpoint per Master Prompt Section 6. Not dependent on the
Orchestrator -- callable and testable on its own.
"""

from fastapi import APIRouter

from app.models.routing import RoutingRequest, RoutingResult
from app.agents.routing_agent import route_complaint

router = APIRouter(prefix="/api/v1/ai", tags=["routing"])


@router.post("/routing", response_model=RoutingResult)
def routing_endpoint(request: RoutingRequest) -> RoutingResult:
    return route_complaint(request)
