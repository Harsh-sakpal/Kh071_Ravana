"""
CivicFlow AI - Escalation Agent Unit Tests
=============================================

Comprehensive test suite for Agent 4: Escalation Agent (ESCALATION_AGENT),
covering all 5 escalation tiers, officer reassignment selection, SLA penalty calculations,
notification draft generation, audit action logging, and HTTP REST endpoint.
"""

from fastapi.testclient import TestClient
from app.main import app
from app.agents.escalation_agent import evaluate_escalation
from app.models.escalation import EscalationRequest, OfficerCandidate

client = TestClient(app)


def test_tier_0_nominal():
    req = EscalationRequest(
        ticket_id="TCK-NOMINAL-01",
        department_code="SAN",
        priority="LOW",
        sla_status="WITHIN_SLA",
        sla_tier="ON_TRACK",
        sla_ratio=0.30,
        breach_probability=0.15,
        is_breached=False,
    )
    res = evaluate_escalation(req)

    assert res.status == "SUCCESS"
    assert res.escalation_tier == "TIER_0_NOMINAL"
    assert res.action_required == "NONE"
    assert res.target_role == "ASSIGNED_OFFICER"
    assert res.penalty_score == 0.0
    assert len(res.audit_trace) > 0


def test_tier_1_supervisor_alert():
    req = EscalationRequest(
        ticket_id="TCK-WARN-01",
        department_code="RDW",
        priority="MEDIUM",
        sla_status="NEAR_BREACH",
        sla_tier="NEAR_BREACH",
        sla_ratio=0.80,
        breach_probability=0.72,
        is_breached=False,
    )
    res = evaluate_escalation(req)

    assert res.status == "SUCCESS"
    assert res.escalation_tier == "TIER_1_SUPERVISOR_ALERT"
    assert res.action_required == "SUPERVISOR_NOTIFIED"
    assert res.target_role == "SUPERVISOR"
    assert "SLA WARNING NOTIFICATION" in res.notification_draft


def test_tier_2_pre_breach_reassignment():
    req = EscalationRequest(
        ticket_id="TCK-REASSIGN-01",
        department_code="WTR",
        priority="HIGH",
        sla_status="NEAR_BREACH",
        sla_tier="CRITICAL_RISK",
        sla_ratio=0.92,
        breach_probability=0.88,
        is_breached=False,
    )
    res = evaluate_escalation(req)

    assert res.status == "SUCCESS"
    assert res.escalation_tier == "TIER_2_PRE_BREACH_REASSIGNMENT"
    assert res.action_required == "REASSIGNMENT_RECOMMENDED"
    assert res.target_role == "REASSIGNED_OFFICER"
    assert res.recommended_officer_id is not None
    assert "PRE-BREACH REASSIGNMENT DIRECTIVE" in res.notification_draft


def test_tier_3_department_head_escalation():
    req = EscalationRequest(
        ticket_id="TCK-BREACHED-01",
        department_code="ELE",
        priority="HIGH",
        sla_status="BREACHED",
        sla_tier="BREACHED",
        sla_ratio=1.20,
        breach_probability=1.0,
        is_breached=True,
    )
    res = evaluate_escalation(req)

    assert res.status == "SUCCESS"
    assert res.escalation_tier == "TIER_3_DEPARTMENT_HEAD_ESCALATION"
    assert res.action_required == "EXEC_BOARD_ESCALATED"
    assert res.target_role == "DEPARTMENT_HEAD"
    assert res.penalty_score > 0.0
    assert "DEPARTMENT HEAD ESCALATION NOTICE" in res.notification_draft


def test_tier_4_emergency_override_safety():
    req = EscalationRequest(
        ticket_id="TCK-EMERGENCY-01",
        department_code="ELE",
        priority="CRITICAL",
        complaint_text="Live sparking wire near school",
        sla_status="BREACHED",
        sla_tier="BREACHED",
        sla_ratio=1.10,
        breach_probability=1.0,
        is_breached=True,
        is_safety_override=True,
    )
    res = evaluate_escalation(req)

    assert res.status == "SUCCESS"
    assert res.escalation_tier == "TIER_4_EMERGENCY_OVERRIDE"
    assert res.action_required == "EMERGENCY_DISPATCH_TRIGGERED"
    assert res.target_role == "EMERGENCY_RESPONSE_UNIT"
    assert "EMERGENCY DISPATCH ALERT" in res.notification_draft


def test_tier_4_emergency_high_affected_citizens():
    req = EscalationRequest(
        ticket_id="TCK-MASS-FLOOD",
        department_code="WTR",
        priority="CRITICAL",
        complaint_text="Water main burst flooding avenue",
        sla_status="BREACHED",
        sla_tier="BREACHED",
        sla_ratio=1.05,
        breach_probability=1.0,
        is_breached=True,
        affected_count=350,
    )
    res = evaluate_escalation(req)

    assert res.status == "SUCCESS"
    assert res.escalation_tier == "TIER_4_EMERGENCY_OVERRIDE"
    assert res.action_required == "EMERGENCY_DISPATCH_TRIGGERED"


def test_custom_officer_reassignment_candidate_selection():
    candidates = [
        OfficerCandidate(officer_id="OFF-BUSY-99", name="Busy Officer", active_tickets=5, is_available=True),
        OfficerCandidate(officer_id="OFF-FREE-01", name="Free Officer", active_tickets=0, is_available=True),
        OfficerCandidate(officer_id="OFF-OFFSHIFT", name="Offshift Officer", active_tickets=0, is_available=False),
    ]
    req = EscalationRequest(
        ticket_id="TCK-CUSTOM-REASSIGN",
        department_code="RDW",
        sla_ratio=0.95,
        candidate_officers=candidates,
    )
    res = evaluate_escalation(req)

    assert res.status == "SUCCESS"
    assert res.recommended_officer_id == "OFF-FREE-01"
    assert res.recommended_officer_name == "Free Officer"


def test_escalation_http_api_endpoint():
    payload = {
        "ticket_id": "TCK-HTTP-99",
        "department_code": "SAN",
        "priority": "HIGH",
        "sla_status": "NEAR_BREACH",
        "sla_tier": "CRITICAL_RISK",
        "sla_ratio": 0.95,
        "breach_probability": 0.88,
        "is_breached": False,
    }
    response = client.post("/api/v1/ai/escalate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["escalation_tier"] == "TIER_2_PRE_BREACH_REASSIGNMENT"
    assert data["action_required"] == "REASSIGNMENT_RECOMMENDED"
    assert data["recommended_officer_id"] is not None
    assert len(data["audit_trace"]) > 0
