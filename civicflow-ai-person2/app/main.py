"""
CivicFlow AI - Person-2 AI Microservice
==========================================

Core production REST services for Routing Agent, DIPEX Priority Agent,
SLA Monitoring Agent, and Escalation Agent.
"""

from fastapi import FastAPI

from app.api.routing import router as routing_router
from app.api.priority import router as priority_router
from app.api.sla_monitoring import router as sla_monitoring_router
from app.api.escalation import router as escalation_router
from app.api.orchestrator import router as orchestrator_router

app = FastAPI(
    title="CivicFlow AI - Person 2 AI Microservice",
    description="Production REST endpoints for Routing, DIPEX Priority, SLA Monitoring, Escalation, and DAG Orchestrator.",
    version="1.1.0",
)

app.include_router(routing_router)
app.include_router(priority_router)
app.include_router(sla_monitoring_router)
app.include_router(escalation_router)
app.include_router(orchestrator_router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "civicflow-ai-person2",
        "version": "1.1.0",
        "active_agents": [
            "ROUTING_AGENT",
            "PRIORITY_AGENT",
            "SLA_MONITORING_AGENT",
            "ESCALATION_AGENT",
            "ORCHESTRATOR_AGENT",
        ]
    }


