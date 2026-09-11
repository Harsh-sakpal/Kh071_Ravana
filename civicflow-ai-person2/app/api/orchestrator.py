"""
CivicFlow AI - Orchestrator REST Endpoint
===========================================

Exposes the POST /api/v1/ai/pipeline endpoint for asynchronous DAG complaint analysis.
"""

from fastapi import APIRouter, HTTPException

from app.models.orchestrator import PipelineRequest, PipelineResult
from app.orchestrator.pipeline import CivicFlowPipeline

router = APIRouter(prefix="/api/v1/ai", tags=["Pipeline Orchestrator"])
pipeline_runner = CivicFlowPipeline()


@router.post(
    "/pipeline",
    response_model=PipelineResult,
    summary="Execute Hybrid Complaint-Analysis DAG Pipeline",
    description=(
        "Executes Routing, Priority, SLA Monitoring, and Escalation agents asynchronously "
        "using a DAG workflow with shared context, parallel execution, and conditional logic."
    ),
)
async def execute_pipeline(request: PipelineRequest) -> PipelineResult:
    """Asynchronous entrypoint for end-to-end civic complaint orchestration."""
    try:
        result = await pipeline_runner.run(request)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"DAG Orchestrator pipeline execution failed: {str(exc)}",
        )
