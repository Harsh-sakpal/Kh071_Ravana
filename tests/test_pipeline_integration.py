"""
CivicFlow AI — Pipeline Integration Tests
=============================================

End-to-end tests that validate the full 8-agent pipeline:
  - Person 1 (CivicFlowAgent) → Bridge → Person 2 (Routing → Priority → SLA → Escalation)
  - All 15 domains (7 aliased + 8 direct) route correctly
  - Taxonomy contract enforcement
  - Confidence threshold validation

These tests use Person 2 agents directly (bypassing Person 1 LLM dependency)
to verify the operational pipeline is structurally sound.
"""

import sys
import os
import pytest
from datetime import datetime, timezone

# Ensure project root and Person 2 root are on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2_ROOT = os.path.join(PROJECT_ROOT, "civicflow-ai-person2")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, P2_ROOT)


# ──────────────────────────────────────────────────────────────────────────
# Taxonomy Bridge Tests
# ──────────────────────────────────────────────────────────────────────────

class TestTaxonomyBridge:
    """Validates the Person 1 → Person 2 domain alias normalization layer."""

    def test_all_aliases_resolve_to_valid_p2_domains(self):
        from app.config.taxonomy import DOMAIN_ALIASES, TAXONOMY
        for p1_name, p2_name in DOMAIN_ALIASES.items():
            assert p2_name in TAXONOMY, (
                f"Alias '{p1_name}' → '{p2_name}' does not map to a valid P2 domain"
            )

    def test_normalize_identity_for_canonical_domains(self):
        from app.config.taxonomy import normalize_domain_name, TAXONOMY
        for domain in TAXONOMY.keys():
            assert normalize_domain_name(domain) == domain

    def test_normalize_aliases(self):
        from app.config.taxonomy import normalize_domain_name
        assert normalize_domain_name("STREET_LIGHTING") == "STREET_LIGHTING_ELECTRICAL"
        assert normalize_domain_name("GARDENS_TREES") == "GARDENS_GREENERY"
        assert normalize_domain_name("PUBLIC_HEALTH") == "PUBLIC_HEALTH_SANITATION"
        assert normalize_domain_name("TRAFFIC_PARKING") == "TRAFFIC_TRANSPORT_PARKING"
        assert normalize_domain_name("PROPERTY_TAX") == "PROPERTY_MUNICIPAL_TAX"
        assert normalize_domain_name("LICENSES_PERMISSIONS") == "LICENSING_PERMISSIONS"
        assert normalize_domain_name("FIRE_DISASTER_SAFETY") == "FIRE_DISASTER_EMERGENCY"

    def test_is_valid_domain_accepts_aliases(self):
        from app.config.taxonomy import is_valid_domain
        assert is_valid_domain("STREET_LIGHTING") is True
        assert is_valid_domain("GARDENS_TREES") is True
        assert is_valid_domain("FAKE_DOMAIN") is False

    def test_subcategory_lookup_via_alias(self):
        from app.config.taxonomy import all_subcategories
        subs = all_subcategories("STREET_LIGHTING")
        assert "STREETLIGHT_OUTAGE" in subs
        assert "DARK_STREET" in subs

    def test_db_mapping_resolves_aliases(self):
        from app.config.db_mapping import department_for_domain
        assert department_for_domain("STREET_LIGHTING") == "ELE"
        assert department_for_domain("GARDENS_TREES") == "PRK"
        assert department_for_domain("PUBLIC_HEALTH") == "SAN"
        assert department_for_domain("TRAFFIC_PARKING") == "RDW"

    def test_all_15_p2_domains_have_department_mapping(self):
        from app.config.taxonomy import TAXONOMY
        from app.config.db_mapping import department_for_domain
        for domain in TAXONOMY.keys():
            dept = department_for_domain(domain)
            assert dept is not None, f"Domain '{domain}' has no department mapping"
            assert dept in {"SAN", "RDW", "ELE", "WTR", "PRK"}, (
                f"Domain '{domain}' maps to unknown department '{dept}'"
            )


# ──────────────────────────────────────────────────────────────────────────
# Routing Agent Integration Tests
# ──────────────────────────────────────────────────────────────────────────

class TestRoutingAgent:
    """Tests Person 2 Routing Agent with Person 1-style inputs."""

    def test_route_with_p1_domain_alias(self):
        from app.agents.routing_agent import route_complaint
        from app.models.routing import RoutingRequest

        req = RoutingRequest(
            complaint_text="The streetlights on MG Road have been off for 3 days",
            existing_domain="STREET_LIGHTING",
        )
        result = route_complaint(req)
        assert result.status in ("SUCCESS", "LOW_CONFIDENCE")
        assert result.department_code == "ELE"

    def test_route_with_canonical_domain(self):
        from app.agents.routing_agent import route_complaint
        from app.models.routing import RoutingRequest

        req = RoutingRequest(
            complaint_text="Garbage not collected from our street for a week",
            existing_domain="SOLID_WASTE",
            existing_subcategory="GARBAGE_NOT_COLLECTED",
        )
        result = route_complaint(req)
        assert result.status == "SUCCESS"
        assert result.domain == "SOLID_WASTE"
        assert result.department_code == "SAN"
        assert result.confidence >= 0.5

    def test_route_without_upstream_domain(self):
        from app.agents.routing_agent import route_complaint
        from app.models.routing import RoutingRequest

        req = RoutingRequest(
            complaint_text="pothole on main road causing accidents",
        )
        result = route_complaint(req)
        assert result.status in ("SUCCESS", "LOW_CONFIDENCE", "UNRESOLVED")


# ──────────────────────────────────────────────────────────────────────────
# Priority Agent Integration Tests
# ──────────────────────────────────────────────────────────────────────────

class TestPriorityAgent:
    """Tests Person 2 Priority Agent with routed data."""

    def test_critical_safety_override(self):
        from app.agents.priority_agent import evaluate_priority
        from app.models.priority import PriorityRequest

        req = PriorityRequest(
            complaint_text="Exposed electrical wire sparking near school playground causing fire hazard",
            domain="STREET_LIGHTING_ELECTRICAL",
            subcategory="EXPOSED_ELECTRICAL_WIRE",
        )
        result = evaluate_priority(req)
        assert result.priority == "CRITICAL"
        assert result.override_applied is True
        assert result.sla_target_hours == 6

    def test_normal_priority_scoring(self):
        from app.agents.priority_agent import evaluate_priority
        from app.models.priority import PriorityRequest

        req = PriorityRequest(
            complaint_text="Park bench is broken in the garden area",
            domain="GARDENS_GREENERY",
            subcategory="DAMAGED_PARK_EQUIPMENT",
        )
        result = evaluate_priority(req)
        assert result.priority in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert 0.0 <= result.priority_score <= 1.0
        assert result.factors is not None


# ──────────────────────────────────────────────────────────────────────────
# SLA Monitoring Agent Integration Tests
# ──────────────────────────────────────────────────────────────────────────

class TestSLAMonitoringAgent:
    """Tests Person 2 SLA Monitoring Agent."""

    def test_new_ticket_on_track(self):
        from app.agents.sla_monitoring_agent import evaluate_sla_status
        from app.models.sla_monitoring import SLAMonitorRequest

        now = datetime.now(timezone.utc)
        req = SLAMonitorRequest(
            ticket_id="TEST-SLA-001",
            priority="MEDIUM",
            status="ASSIGNED",
            created_at=now,
            current_time=now,
        )
        result = evaluate_sla_status(req)
        assert result.sla_status == "WITHIN_SLA"
        assert result.sla_tier == "ON_TRACK"
        assert result.is_breached is False

    def test_resolved_ticket_exempt(self):
        from app.agents.sla_monitoring_agent import evaluate_sla_status
        from app.models.sla_monitoring import SLAMonitorRequest

        now = datetime.now(timezone.utc)
        req = SLAMonitorRequest(
            ticket_id="TEST-SLA-002",
            priority="HIGH",
            status="RESOLVED",
            created_at=now,
            current_time=now,
        )
        result = evaluate_sla_status(req)
        assert result.sla_status == "EXEMPT"


# ──────────────────────────────────────────────────────────────────────────
# Escalation Agent Integration Tests
# ──────────────────────────────────────────────────────────────────────────

class TestEscalationAgent:
    """Tests Person 2 Escalation Agent."""

    def test_nominal_no_escalation(self):
        from app.agents.escalation_agent import evaluate_escalation
        from app.models.escalation import EscalationRequest

        req = EscalationRequest(
            ticket_id="TEST-ESC-001",
            department_code="SAN",
            priority="MEDIUM",
            sla_status="WITHIN_SLA",
            sla_ratio=0.30,
            breach_probability=0.15,
        )
        result = evaluate_escalation(req)
        assert result.escalation_tier == "TIER_0_NOMINAL"
        assert result.action_required == "NONE"

    def test_breach_triggers_tier3(self):
        from app.agents.escalation_agent import evaluate_escalation
        from app.models.escalation import EscalationRequest

        req = EscalationRequest(
            ticket_id="TEST-ESC-002",
            department_code="WTR",
            priority="HIGH",
            sla_status="BREACHED",
            sla_ratio=1.2,
            breach_probability=1.0,
            is_breached=True,
        )
        result = evaluate_escalation(req)
        assert result.escalation_tier in (
            "TIER_3_DEPARTMENT_HEAD_ESCALATION",
            "TIER_4_EMERGENCY_OVERRIDE",
        )
        assert result.penalty_score > 0

    def test_emergency_override_tier4(self):
        from app.agents.escalation_agent import evaluate_escalation
        from app.models.escalation import EscalationRequest

        req = EscalationRequest(
            ticket_id="TEST-ESC-003",
            department_code="ELE",
            priority="CRITICAL",
            sla_status="BREACHED",
            sla_ratio=2.0,
            breach_probability=1.0,
            is_breached=True,
            is_safety_override=True,
            affected_count=500,
        )
        result = evaluate_escalation(req)
        assert result.escalation_tier == "TIER_4_EMERGENCY_OVERRIDE"
        assert result.action_required == "EMERGENCY_DISPATCH_TRIGGERED"


# ──────────────────────────────────────────────────────────────────────────
# Full Pipeline Chain Test (P2 agents only — no LLM dependency)
# ──────────────────────────────────────────────────────────────────────────

class TestFullPipelineChain:
    """
    Tests the complete P2 chain: Routing → Priority → SLA → Escalation
    to verify all agents produce compatible outputs.
    """

    def test_end_to_end_chain(self):
        from app.agents.routing_agent import route_complaint
        from app.agents.priority_agent import evaluate_priority
        from app.agents.sla_monitoring_agent import evaluate_sla_status
        from app.agents.escalation_agent import evaluate_escalation
        from app.models.routing import RoutingRequest
        from app.models.priority import PriorityRequest
        from app.models.sla_monitoring import SLAMonitorRequest
        from app.models.escalation import EscalationRequest

        # Step 1: Route
        routing_req = RoutingRequest(
            complaint_text="Water pipeline leaking heavily on main road for 3 days",
            existing_domain="WATER_SUPPLY",
            existing_subcategory="PIPELINE_LEAK",
        )
        routing_result = route_complaint(routing_req)
        assert routing_result.status == "SUCCESS"

        # Step 2: Priority
        priority_req = PriorityRequest(
            complaint_text="Water pipeline leaking heavily on main road for 3 days",
            domain=routing_result.domain,
            subcategory=routing_result.subcategory,
            category_code=routing_result.category_code,
            duration_days=3.0,
        )
        priority_result = evaluate_priority(priority_req)
        assert priority_result.priority in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

        # Step 3: SLA
        now = datetime.now(timezone.utc)
        sla_req = SLAMonitorRequest(
            ticket_id="CHAIN-TEST-001",
            priority=priority_result.priority,
            category_code=routing_result.category_code,
            subcategory=routing_result.subcategory,
            status="ASSIGNED",
            created_at=now,
            current_time=now,
            sla_target_hours=priority_result.sla_target_hours,
        )
        sla_result = evaluate_sla_status(sla_req)
        assert sla_result.status == "SUCCESS"

        # Step 4: Escalation
        esc_req = EscalationRequest(
            ticket_id="CHAIN-TEST-001",
            complaint_text="Water pipeline leaking heavily on main road for 3 days",
            department_code=routing_result.department_code or "WTR",
            priority=priority_result.priority,
            sla_status=sla_result.sla_status,
            sla_ratio=sla_result.sla_ratio,
            sla_tier=sla_result.sla_tier,
            is_breached=sla_result.is_breached,
            breach_probability=sla_result.breach_probability,
        )
        esc_result = evaluate_escalation(esc_req)
        assert esc_result.status == "SUCCESS"
        assert esc_result.escalation_tier is not None

        # Verify chain consistency
        assert routing_result.department_code is not None
        assert priority_result.sla_target_hours > 0
        assert sla_result.sla_target_hours > 0
