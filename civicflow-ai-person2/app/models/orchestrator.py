"""
CivicFlow AI - Pipeline Orchestrator Pydantic Models & Shared Context
========================================================================

Defines input/output schemas, shared pipeline context, and node execution status
models for the lightweight asynchronous DAG orchestrator in Person 2 microservice.
"""

from datetime import datetime
import time
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from app.models.routing import RoutingResult
from app.models.priority import PriorityResult
from app.models.sla_monitoring import SLAMonitorResult
from app.models.escalation import EscalationResult, OfficerCandidate


class PipelineRequest(BaseModel):
    """Unified entrypoint request for the hybrid complaint analysis DAG pipeline."""

    complaint_id: Optional[str] = Field(
        default=None, description="Optional tracking UUID/ID for the complaint."
    )
    complaint_text: str = Field(
        ..., min_length=3, description="Raw or lightly cleaned complaint text description."
    )
    existing_domain: Optional[str] = Field(
        default=None, description="Optional domain pre-classified upstream."
    )
    existing_subcategory: Optional[str] = Field(
        default=None, description="Optional subcategory pre-classified upstream."
    )
    location_context: Optional[str] = Field(
        default=None, description="Optional location description notes (e.g. near school)."
    )
    affected_count: Optional[int] = Field(
        default=None, ge=0, description="Estimated number of affected citizens."
    )
    duration_days: Optional[float] = Field(
        default=None, ge=0.0, description="Duration in days the issue has persisted."
    )
    recurrence_count: Optional[int] = Field(
        default=None, ge=0, description="Count of past repeated complaints in same area."
    )
    created_at: Optional[datetime] = Field(
        default=None, description="Complaint creation timestamp."
    )
    officer_workload_count: Optional[int] = Field(
        default=None, ge=0, description="Active tickets count for assigned officer."
    )
    department_queue_count: Optional[int] = Field(
        default=None, ge=0, description="Unassigned active tickets in department queue."
    )
    is_sensitive_location: Optional[bool] = Field(
        default=False, description="True if near a school, hospital, or sensitive landmark."
    )
    historical_breach_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Historical category breach probability (0.0 to 1.0)."
    )
    current_officer_id: Optional[str] = Field(
        default="OFF-ASSIGNED-01", description="Currently assigned officer identifier."
    )
    candidate_officers: Optional[List[OfficerCandidate]] = Field(
        default=None, description="Optional list of available candidate officers for reassignment."
    )


class NodeExecutionStatus(BaseModel):
    """Tracks state and performance metrics of individual DAG pipeline nodes."""

    node_id: str = Field(..., description="Unique node identifier in the DAG.")
    status: str = Field(
        ..., description="Node state: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED."
    )
    duration_ms: float = Field(default=0.0, ge=0.0, description="Execution time in milliseconds.")
    error_message: Optional[str] = Field(default=None, description="Error message if node failed.")
    skip_reason: Optional[str] = Field(
        default=None, description="Reason node execution was skipped if condition false."
    )


class PipelineContext:
    """Thread/task-safe container for shared state throughout DAG execution lifecycle."""

    def __init__(self, request: PipelineRequest, pipeline_id: Optional[str] = None):
        self.pipeline_id: str = pipeline_id or f"PIPE-{uuid.uuid4().hex[:8].upper()}"
        self.request: PipelineRequest = request
        self.outputs: Dict[str, Any] = {}
        self.node_statuses: Dict[str, NodeExecutionStatus] = {}
        self.start_time: float = time.perf_counter()
        self.end_time: Optional[float] = None
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def set_output(self, node_id: str, value: Any) -> None:
        self.outputs[node_id] = value

    def get_output(self, node_id: str, default: Any = None) -> Any:
        return self.outputs.get(node_id, default)

    def set_node_status(
        self,
        node_id: str,
        status: str,
        duration_ms: float = 0.0,
        error_message: Optional[str] = None,
        skip_reason: Optional[str] = None,
    ) -> None:
        self.node_statuses[node_id] = NodeExecutionStatus(
            node_id=node_id,
            status=status,
            duration_ms=round(duration_ms, 2),
            error_message=error_message,
            skip_reason=skip_reason,
        )

    def total_duration_ms(self) -> float:
        end = self.end_time or time.perf_counter()
        return round((end - self.start_time) * 1000, 2)


class PipelineResult(BaseModel):
    """Aggregated response returned by the DAG Orchestrator after pipeline execution."""

    status: str = Field(..., description="Overall status: SUCCESS, PARTIAL_SUCCESS, or FAILED.")
    pipeline_id: str = Field(..., description="Unique DAG execution tracking ID.")
    complaint_id: Optional[str] = Field(default=None, description="Complaint identifier.")
    execution_time_ms: float = Field(..., ge=0.0, description="Total pipeline execution latency in ms.")
    dag_summary: List[NodeExecutionStatus] = Field(
        default_factory=list, description="Per-node execution summary breakdown."
    )
    routing: Optional[RoutingResult] = Field(
        default=None, description="Routing Agent output if executed."
    )
    priority: Optional[PriorityResult] = Field(
        default=None, description="Priority Agent output if executed."
    )
    sla: Optional[SLAMonitorResult] = Field(
        default=None, description="SLA Monitoring Agent output if executed."
    )
    escalation: Optional[EscalationResult] = Field(
        default=None, description="Escalation Agent output if executed."
    )
    warnings: List[str] = Field(
        default_factory=list, description="List of warnings accumulated during execution."
    )
