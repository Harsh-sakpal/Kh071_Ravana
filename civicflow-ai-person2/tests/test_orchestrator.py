"""
CivicFlow AI - Orchestrator & DAG Pipeline Test Suite
======================================================

Tests DAG graph validation, cycle detection, parallel node execution, shared context,
conditional node skipping, fallback error handling, hybrid pipeline orchestration,
and FastAPI REST endpoint integration.
"""

import asyncio
import time
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.orchestrator import (
    NodeExecutionStatus,
    PipelineContext,
    PipelineRequest,
    PipelineResult,
)
from app.orchestrator.dag import DAGNode, DAGOrchestrator
from app.orchestrator.pipeline import CivicFlowPipeline

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. DAG Engine Core Unit Tests
# ---------------------------------------------------------------------------

def test_dag_cycle_detection():
    async def _run():
        orchestrator = DAGOrchestrator()
        orchestrator.add_node(DAGNode("node1", handler=lambda ctx: None, depends_on=["node2"]))
        orchestrator.add_node(DAGNode("node2", handler=lambda ctx: None, depends_on=["node1"]))

        with pytest.raises(ValueError, match="Cycle detected in DAG"):
            orchestrator.validate_dag()

    asyncio.run(_run())


def test_dag_missing_dependency():
    async def _run():
        orchestrator = DAGOrchestrator()
        orchestrator.add_node(DAGNode("node1", handler=lambda ctx: None, depends_on=["non_existent"]))

        with pytest.raises(ValueError, match="depends on non-existent node"):
            orchestrator.validate_dag()

    asyncio.run(_run())


def test_parallel_node_execution():
    """Verify that independent nodes run concurrently, saving overall execution time."""
    async def _run():
        orchestrator = DAGOrchestrator()
        execution_times = {}

        async def sleep_node(node_name: str, delay: float):
            start = time.perf_counter()
            await asyncio.sleep(delay)
            execution_times[node_name] = time.perf_counter() - start
            return node_name

        orchestrator.add_node(DAGNode("base", handler=lambda ctx: sleep_node("base", 0.05)))
        orchestrator.add_node(
            DAGNode("parallel_1", handler=lambda ctx: sleep_node("p1", 0.1), depends_on=["base"])
        )
        orchestrator.add_node(
            DAGNode("parallel_2", handler=lambda ctx: sleep_node("p2", 0.1), depends_on=["base"])
        )

        req = PipelineRequest(complaint_text="Parallel test complaint")
        ctx = PipelineContext(req)

        start_total = time.perf_counter()
        await orchestrator.execute(ctx)
        total_time = time.perf_counter() - start_total

        assert ctx.node_statuses["base"].status == "COMPLETED"
        assert ctx.node_statuses["parallel_1"].status == "COMPLETED"
        assert ctx.node_statuses["parallel_2"].status == "COMPLETED"

        # Parallel nodes delay = 0.1 each. If serial, total would be > 0.25s. If parallel, total < 0.22s.
        assert total_time < 0.22

    asyncio.run(_run())


def test_node_fallback_on_failure():
    async def _run():
        orchestrator = DAGOrchestrator()

        async def failing_handler(ctx: PipelineContext):
            raise RuntimeError("Primary database connection error")

        async def fallback_handler(ctx: PipelineContext, exc: Exception):
            return {"recovered": True, "error": str(exc)}

        orchestrator.add_node(
            DAGNode(
                node_id="fallback_test",
                handler=failing_handler,
                fallback=fallback_handler,
                retry_count=1,
            )
        )

        req = PipelineRequest(complaint_text="Fallback test")
        ctx = PipelineContext(req)
        await orchestrator.execute(ctx)

        assert ctx.node_statuses["fallback_test"].status == "COMPLETED"
        assert "Used fallback" in ctx.node_statuses["fallback_test"].error_message
        assert ctx.get_output("fallback_test") == {"recovered": True, "error": "Primary database connection error"}

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. CivicFlow Hybrid Pipeline Integration Tests
# ---------------------------------------------------------------------------

def test_pipeline_routine_complaint_skips_escalation():
    async def _run():
        pipeline = CivicFlowPipeline()
        req = PipelineRequest(
            complaint_id="CMP-ROUTINE-01",
            complaint_text="Paver block loose on quiet sidewalk garden walkway.",
            existing_domain="GARDENS_GREENERY",
            affected_count=1,
        )

        result = await pipeline.run(req)

        assert result.status == "SUCCESS"
        assert result.complaint_id == "CMP-ROUTINE-01"
        assert result.routing is not None
        assert result.priority is not None
        assert result.sla is not None

        # Escalation should be skipped because priority is LOW/MEDIUM and SLA is within target
        assert result.escalation is None
        escalation_status = next(s for s in result.dag_summary if s.node_id == "escalation")
        assert escalation_status.status == "SKIPPED"
        assert "Condition predicate evaluated to False" in escalation_status.skip_reason

    asyncio.run(_run())


def test_pipeline_critical_complaint_triggers_escalation():
    async def _run():
        pipeline = CivicFlowPipeline()
        req = PipelineRequest(
            complaint_id="CMP-CRIT-99",
            complaint_text="Sparking transformer live exposed wire near fuel station and school.",
            affected_count=100,
            is_sensitive_location=True,
        )

        result = await pipeline.run(req)

        assert result.status == "SUCCESS"
        assert result.routing is not None
        assert result.priority is not None
        assert result.priority.priority == "CRITICAL"
        assert result.priority.override_applied is True

        # Escalation node should run and succeed
        assert result.escalation is not None
        assert result.escalation.status == "SUCCESS"
        assert result.escalation.escalation_tier in (
            "TIER_0_NOMINAL",
            "TIER_1_SUPERVISOR_ALERT",
            "TIER_2_PRE_BREACH_REASSIGNMENT",
            "TIER_3_DEPARTMENT_HEAD_ESCALATION",
            "TIER_4_EMERGENCY_OVERRIDE",
        )
        escalation_status = next(s for s in result.dag_summary if s.node_id == "escalation")
        assert escalation_status.status == "COMPLETED"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. HTTP REST API Endpoint Tests
# ---------------------------------------------------------------------------

def test_health_endpoint_registers_orchestrator():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ORCHESTRATOR_AGENT" in data["active_agents"]


def test_pipeline_http_endpoint_routine():
    payload = {
        "complaint_id": "TCK-HTTP-01",
        "complaint_text": "Garbage has not been collected for three days outside society.",
        "affected_count": 15,
    }
    response = client.post("/api/v1/ai/pipeline", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "SUCCESS"
    assert data["complaint_id"] == "TCK-HTTP-01"
    assert data["execution_time_ms"] > 0
    assert len(data["dag_summary"]) == 4

    assert data["routing"]["domain"] == "SOLID_WASTE"
    assert data["routing"]["subcategory"] == "GARBAGE_NOT_COLLECTED"
    assert data["priority"]["priority"] in ("MEDIUM", "HIGH")
    assert data["sla"]["sla_status"] == "WITHIN_SLA"


def test_pipeline_http_endpoint_urgent():
    payload = {
        "complaint_id": "TCK-HTTP-URGENT",
        "complaint_text": "Building wall collapse hazard near hospital emergency ward.",
        "affected_count": 50,
        "is_sensitive_location": True,
    }
    response = client.post("/api/v1/ai/pipeline", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "SUCCESS"
    assert data["priority"]["priority"] == "CRITICAL"
    assert data["priority"]["override_applied"] is True
    assert data["escalation"] is not None
    assert "notification_draft" in data["escalation"]
