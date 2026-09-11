"""
CivicFlow AI - Advanced SLA Monitoring Agent Unit & Integration Tests
================================================-----------------------

Tests category SLA rules, sensitive location SLA discounts, 6-factor breach probability,
multi-stage risk tiers (ON_TRACK, PROGRESSING, NEAR_BREACH, CRITICAL_RISK, BREACHED, EXEMPT),
and HTTP REST endpoint.
"""

from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

from app.main import app
from app.models.sla_monitoring import SLAMonitorRequest
from app.agents.sla_monitoring_agent import evaluate_sla_status

client = TestClient(app)


def test_1_within_sla_on_track():
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=4)
    req = SLAMonitorRequest(
        ticket_id="TCK-1001",
        created_at=created,
        current_time=now,
        priority="HIGH",
        status="IN_PROGRESS",
    )
    res = evaluate_sla_status(req)
    assert res.status == "SUCCESS"
    assert res.sla_status == "WITHIN_SLA"
    assert res.sla_tier == "ON_TRACK"
    assert res.elapsed_hours == 4.0
    assert res.remaining_hours == 20.0
    assert res.sla_target_hours == 24
    assert res.is_warning_triggered is False
    assert res.is_breached is False
    assert res.recommendation_action == "MAINTAIN_NORMAL_MONITORING"


def test_2_within_sla_progressing():
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=14)
    req = SLAMonitorRequest(
        ticket_id="TCK-1002",
        created_at=created,
        current_time=now,
        priority="HIGH",
        status="IN_PROGRESS",
    )
    res = evaluate_sla_status(req)
    assert res.sla_status == "WITHIN_SLA"
    assert res.sla_tier == "PROGRESSING"
    assert res.recommendation_action == "MONITOR_PROGRESS"


def test_3_near_breach_warning():
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=20)
    req = SLAMonitorRequest(
        ticket_id="TCK-1003",
        created_at=created,
        current_time=now,
        priority="HIGH",
        status="IN_PROGRESS",
        officer_workload_count=6,
    )
    res = evaluate_sla_status(req)
    assert res.sla_status == "NEAR_BREACH"
    assert res.sla_tier == "NEAR_BREACH"
    assert res.is_warning_triggered is True
    assert res.is_breached is False
    assert res.recommendation_action == "TRIGGER_NEAR_BREACH_WARNING"


def test_4_critical_risk_pre_breach():
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=22.5)
    req = SLAMonitorRequest(
        ticket_id="TCK-1004",
        created_at=created,
        current_time=now,
        priority="HIGH",
        status="IN_PROGRESS",
    )
    res = evaluate_sla_status(req)
    assert res.sla_status == "NEAR_BREACH"
    assert res.sla_tier == "CRITICAL_RISK"
    assert res.is_warning_triggered is True
    assert res.recommendation_action == "PREPARE_PRE_BREACH_ESCALATION"


def test_5_breached_scenario():
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=8)
    req = SLAMonitorRequest(
        ticket_id="TCK-1005",
        created_at=created,
        current_time=now,
        priority="CRITICAL",  # 6h SLA target
        status="ASSIGNED",
    )
    res = evaluate_sla_status(req)
    assert res.sla_status == "BREACHED"
    assert res.sla_tier == "BREACHED"
    assert res.elapsed_hours == 8.0
    assert res.remaining_hours == 0.0
    assert res.is_breached is True
    assert res.is_warning_triggered is True
    assert res.breach_probability == 1.00
    assert res.recommendation_action == "IMMEDIATE_ESCALATION_REQUIRED"


def test_6_category_sla_rule_resolution():
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=2)
    req = SLAMonitorRequest(
        created_at=created,
        current_time=now,
        category_code="WTR-LEAK",
    )
    res = evaluate_sla_status(req)
    assert res.sla_target_hours == 6


def test_7_sensitive_location_sla_reduction():
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=2)
    req = SLAMonitorRequest(
        created_at=created,
        current_time=now,
        priority="HIGH",  # Default 24h
        is_sensitive_location=True,  # 25% discount -> 18h
    )
    res = evaluate_sla_status(req)
    assert res.sla_target_hours == 18
    assert any("Sensitive location discount" in w for w in res.warnings)


def test_8_exempt_status():
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=10)
    req = SLAMonitorRequest(
        ticket_id="TCK-1008",
        created_at=created,
        current_time=now,
        priority="CRITICAL",
        status="RESOLVED",
    )
    res = evaluate_sla_status(req)
    assert res.sla_status == "EXEMPT"
    assert res.sla_tier == "EXEMPT"
    assert res.is_warning_triggered is False
    assert res.is_breached is False
    assert res.breach_probability == 0.0
    assert res.recommendation_action == "TICKET_RESOLVED_NO_ACTION_REQUIRED"


def test_9_sla_factors_breakdown():
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=18)
    req = SLAMonitorRequest(
        created_at=created,
        current_time=now,
        priority="HIGH",
        subcategory="WATER_MAIN_LEAK",
        officer_workload_count=7,
        department_queue_count=25,
        is_sensitive_location=True,
    )
    res = evaluate_sla_status(req)
    assert res.sla_factors.complexity >= 0.80
    assert res.sla_factors.officer_workload >= 0.70
    assert res.sla_factors.location_urgency >= 0.80
    assert "SLA status is" in res.explanation


def test_10_sla_monitor_endpoint_http():
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=21)
    payload = {
        "ticket_id": "TCK-2026-9999",
        "created_at": created.isoformat(),
        "current_time": now.isoformat(),
        "priority": "HIGH",
        "status": "IN_PROGRESS",
        "is_sensitive_location": False,
    }
    response = client.post("/api/v1/ai/sla-monitor", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["sla_status"] == "NEAR_BREACH"
    assert data["sla_tier"] == "NEAR_BREACH"
    assert data["remaining_hours"] == 3.0
    assert data["is_warning_triggered"] is True
    assert "sla_factors" in data
    assert data["recommendation_action"] == "TRIGGER_NEAR_BREACH_WARNING"
