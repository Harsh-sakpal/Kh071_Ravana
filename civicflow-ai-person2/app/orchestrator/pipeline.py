"""
CivicFlow AI - Hybrid Complaint-Analysis Pipeline
====================================================

Constructs and executes the multi-agent DAG pipeline connecting:
1. RoutingAgent (Domain/Subcategory/Department classification)
2. PriorityAgent & SLAMonitoringAgent (Concurrent execution)
3. EscalationAgent (Conditional execution based on priority/SLA breach risk)
"""

import logging
from typing import Optional

from app.agents.routing_agent import route_complaint
from app.agents.priority_agent import evaluate_priority
from app.agents.sla_monitoring_agent import evaluate_sla_status
from app.agents.escalation_agent import evaluate_escalation

from app.models.routing import RoutingRequest, RoutingResult
from app.models.priority import PriorityRequest, PriorityResult
from app.models.sla_monitoring import SLAMonitorRequest, SLAMonitorResult
from app.models.escalation import EscalationRequest, EscalationResult
from app.models.orchestrator import (
    NodeExecutionStatus,
    PipelineContext,
    PipelineRequest,
    PipelineResult,
)
from app.orchestrator.dag import DAGNode, DAGOrchestrator

logger = logging.getLogger(__name__)


class CivicFlowPipeline:
    """Hybrid complaint analysis DAG pipeline manager."""

    def __init__(self):
        self.orchestrator: DAGOrchestrator = self._build_dag()

    def _build_dag(self) -> DAGOrchestrator:
        orchestrator = DAGOrchestrator()

        # Node 1: Routing Agent (Base Node)
        orchestrator.add_node(
            DAGNode(
                node_id="routing",
                handler=self._routing_handler,
                depends_on=[],
                retry_count=1,
            )
        )

        # Node 2: Priority Agent (Depends on Routing)
        orchestrator.add_node(
            DAGNode(
                node_id="priority",
                handler=self._priority_handler,
                depends_on=["routing"],
                retry_count=1,
            )
        )

        # Node 3: SLA Monitoring Agent (Depends on Routing, runs in parallel with Priority)
        orchestrator.add_node(
            DAGNode(
                node_id="sla_monitoring",
                handler=self._sla_handler,
                depends_on=["routing"],
                retry_count=1,
            )
        )

        # Node 4: Escalation Agent (Depends on Priority & SLA, Conditional)
        orchestrator.add_node(
            DAGNode(
                node_id="escalation",
                handler=self._escalation_handler,
                depends_on=["priority", "sla_monitoring"],
                condition=self._escalation_condition,
                retry_count=1,
            )
        )

        return orchestrator

    async def _routing_handler(self, context: PipelineContext) -> RoutingResult:
        req = context.request
        routing_req = RoutingRequest(
            complaint_id=req.complaint_id,
            complaint_text=req.complaint_text,
            existing_domain=req.existing_domain,
            existing_subcategory=req.existing_subcategory,
        )
        result = route_complaint(routing_req)
        return result

    async def _priority_handler(self, context: PipelineContext) -> PriorityResult:
        req = context.request
        routing_res: Optional[RoutingResult] = context.get_output("routing")

        domain = routing_res.domain if routing_res else req.existing_domain
        subcategory = routing_res.subcategory if routing_res else req.existing_subcategory
        category_code = routing_res.category_code if routing_res else None

        priority_req = PriorityRequest(
            complaint_id=req.complaint_id,
            complaint_text=req.complaint_text,
            domain=domain,
            subcategory=subcategory,
            category_code=category_code,
            location_context=req.location_context,
            affected_count=req.affected_count,
            duration_days=req.duration_days,
            recurrence_count=req.recurrence_count,
        )
        result = evaluate_priority(priority_req)
        return result

    async def _sla_handler(self, context: PipelineContext) -> SLAMonitorResult:
        req = context.request
        routing_res: Optional[RoutingResult] = context.get_output("routing")

        category_code = routing_res.category_code if routing_res else None
        subcategory = routing_res.subcategory if routing_res else req.existing_subcategory

        sla_req = SLAMonitorRequest(
            complaint_id=req.complaint_id,
            created_at=req.created_at,
            priority="MEDIUM",
            category_code=category_code,
            subcategory=subcategory,
            officer_workload_count=req.officer_workload_count,
            department_queue_count=req.department_queue_count,
            is_sensitive_location=req.is_sensitive_location or False,
            historical_breach_rate=req.historical_breach_rate,
        )
        result = evaluate_sla_status(sla_req)
        return result

    def _escalation_condition(self, context: PipelineContext) -> bool:
        """Determines if escalation analysis should be executed.

        Escalation runs if:
        - Priority is HIGH or CRITICAL
        - Safety override was applied in Priority Agent
        - SLA status is NEAR_BREACH or BREACHED
        - SLA breach warning was triggered
        """
        priority_res: Optional[PriorityResult] = context.get_output("priority")
        sla_res: Optional[SLAMonitorResult] = context.get_output("sla_monitoring")

        if priority_res:
            if priority_res.priority in ("HIGH", "CRITICAL"):
                return True
            if priority_res.override_applied:
                return True

        if sla_res:
            if sla_res.sla_status in ("NEAR_BREACH", "BREACHED"):
                return True
            if sla_res.is_warning_triggered or sla_res.is_breached:
                return True

        return False

    async def _escalation_handler(self, context: PipelineContext) -> EscalationResult:
        req = context.request
        routing_res: Optional[RoutingResult] = context.get_output("routing")
        priority_res: Optional[PriorityResult] = context.get_output("priority")
        sla_res: Optional[SLAMonitorResult] = context.get_output("sla_monitoring")

        dept_code = routing_res.department_code if (routing_res and routing_res.department_code) else "SAN"
        prio = priority_res.priority if priority_res else "MEDIUM"
        sla_status = sla_res.sla_status if sla_res else "WITHIN_SLA"
        sla_tier = sla_res.sla_tier if sla_res else "ON_TRACK"
        sla_ratio = sla_res.sla_ratio if sla_res else 0.0
        breach_prob = sla_res.breach_probability if sla_res else 0.0
        is_breached = sla_res.is_breached if sla_res else False
        is_override = priority_res.override_applied if priority_res else False

        escalation_req = EscalationRequest(
            ticket_id=req.complaint_id or "TCK-DAG-2026",
            complaint_text=req.complaint_text,
            department_code=dept_code,
            priority=prio,
            sla_status=sla_status,
            sla_tier=sla_tier,
            sla_ratio=sla_ratio,
            breach_probability=breach_prob,
            is_breached=is_breached,
            is_safety_override=is_override,
            affected_count=req.affected_count or 0,
            current_officer_id=req.current_officer_id or "OFF-ASSIGNED-01",
            candidate_officers=req.candidate_officers,
        )

        result = evaluate_escalation(escalation_req)
        return result

    async def run(self, request: PipelineRequest) -> PipelineResult:
        """Executes the pipeline request through the DAG orchestrator."""
        context = PipelineContext(request)
        await self.orchestrator.execute(context)

        routing_res: Optional[RoutingResult] = context.get_output("routing")
        priority_res: Optional[PriorityResult] = context.get_output("priority")
        sla_res: Optional[SLAMonitorResult] = context.get_output("sla_monitoring")
        escalation_res: Optional[EscalationResult] = context.get_output("escalation")

        # Accumulate warnings across nodes
        accumulated_warnings = list(context.warnings)
        if routing_res and routing_res.warnings:
            accumulated_warnings.extend(routing_res.warnings)
        if priority_res and priority_res.warnings:
            accumulated_warnings.extend(priority_res.warnings)
        if sla_res and sla_res.warnings:
            accumulated_warnings.extend(sla_res.warnings)
        if escalation_res and escalation_res.warnings:
            accumulated_warnings.extend(escalation_res.warnings)

        dag_summary = list(context.node_statuses.values())

        # Determine overall pipeline status
        routing_status = context.node_statuses.get("routing")
        if routing_status and routing_status.status == "FAILED":
            overall_status = "FAILED"
        elif any(ns.status == "FAILED" for ns in dag_summary):
            overall_status = "PARTIAL_SUCCESS"
        else:
            overall_status = "SUCCESS"

        return PipelineResult(
            status=overall_status,
            pipeline_id=context.pipeline_id,
            complaint_id=request.complaint_id,
            execution_time_ms=context.total_duration_ms(),
            dag_summary=dag_summary,
            routing=routing_res,
            priority=priority_res,
            sla=sla_res,
            escalation=escalation_res,
            warnings=list(dict.fromkeys(accumulated_warnings)),  # Deduplicate while preserving order
        )
