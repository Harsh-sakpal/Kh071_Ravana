"""
Import bridge for Person 2 agent functions, orchestrator, and request/response models.
Centralizes the sys.path manipulation needed for cross-package imports.
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_p2_root = os.path.join(_project_root, "civicflow-ai-person2")

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _p2_root not in sys.path:
    sys.path.insert(0, _p2_root)

# Routing Agent
from app.agents.routing_agent import route_complaint
from app.models.routing import RoutingRequest, RoutingResult

# Priority Agent
from app.agents.priority_agent import evaluate_priority
from app.models.priority import PriorityRequest, PriorityResult

# SLA Monitoring Agent
from app.agents.sla_monitoring_agent import evaluate_sla_status
from app.models.sla_monitoring import SLAMonitorRequest, SLAMonitorResult

# Escalation Agent
from app.agents.escalation_agent import evaluate_escalation
from app.models.escalation import EscalationRequest, EscalationResult

# Orchestrator & Pipeline
from app.orchestrator.dag import DAGNode, DAGOrchestrator
from app.orchestrator.pipeline import CivicFlowPipeline
from app.models.orchestrator import (
    PipelineRequest,
    PipelineResult,
    PipelineContext,
    NodeExecutionStatus,
)

__all__ = [
    "route_complaint", "RoutingRequest", "RoutingResult",
    "evaluate_priority", "PriorityRequest", "PriorityResult",
    "evaluate_sla_status", "SLAMonitorRequest", "SLAMonitorResult",
    "evaluate_escalation", "EscalationRequest", "EscalationResult",
    "DAGNode", "DAGOrchestrator", "CivicFlowPipeline",
    "PipelineRequest", "PipelineResult", "PipelineContext", "NodeExecutionStatus",
]
