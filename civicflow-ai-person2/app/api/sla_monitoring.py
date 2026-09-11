"""
CivicFlow AI - SLA Monitoring Agent API
=========================================

Independent REST endpoint for automated ticket SLA deadline tracking, remaining time,
breach probability, and warning alerts.
"""

from fastapi import APIRouter

from app.models.sla_monitoring import SLAMonitorRequest, SLAMonitorResult
from app.agents.sla_monitoring_agent import evaluate_sla_status

router = APIRouter(prefix="/api/v1/ai", tags=["sla-monitoring"])


@router.post("/sla-monitor", response_model=SLAMonitorResult)
def sla_monitor_endpoint(request: SLAMonitorRequest) -> SLAMonitorResult:
    return evaluate_sla_status(request)
