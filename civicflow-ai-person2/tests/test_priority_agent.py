"""
CivicFlow AI - Priority Agent Unit & Integration Tests (18 DIPEX Test Cases)
================================================----------------===============

Comprehensive test coverage for DIPEX continuous scoring (0.0 - 1.0), 10 factor extractions,
threshold conversions, deterministic safety overrides, category operational baselines,
optional data integrations (SLA/Hotspot/Risk), and FastAPI HTTP contract.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.models.priority import (
    HotspotData,
    PriorityRequest,
    RiskData,
    SLAData,
)
from app.agents.priority_agent import evaluate_priority
from app.config.priority_config import PRIORITY_THRESHOLDS

client = TestClient(app)


# 1. Low-severity complaint
def test_1_low_severity_complaint():
    req = PriorityRequest(
        complaint_text="Park bench repaint request and routine grass trimming in local garden.",
        domain="GARDENS_GREENERY",
        subcategory="PARK_MAINTENANCE",
        affected_count=1,
    )
    res = evaluate_priority(req)
    assert res.priority == "LOW"
    assert res.priority_score < PRIORITY_THRESHOLDS["low_max"]
    assert res.sla_target_hours == 120
    assert 0.0 <= res.priority_score <= 1.0


# 2. Medium complaint
def test_2_medium_complaint():
    req = PriorityRequest(
        complaint_text="Garbage bin is full and requires regular scheduled pickup.",
        domain="SOLID_WASTE",
        subcategory="GARBAGE_NOT_COLLECTED",
        category_code="SAN-GARB",
        affected_count=10,
    )
    res = evaluate_priority(req)
    assert res.priority in ("MEDIUM", "HIGH")
    assert 0.0 <= res.priority_score <= 1.0


# 3. High-severity complaint
def test_3_high_severity_complaint():
    req = PriorityRequest(
        complaint_text="Large deep pothole on major road causing vehicle damage and traffic slowdown.",
        domain="ROADS_FOOTPATHS",
        subcategory="POTHOLE",
        category_code="RDW-POT",
        affected_count=80,
    )
    res = evaluate_priority(req)
    assert res.priority in ("HIGH", "CRITICAL")
    assert res.factors.severity >= 0.50


# 4. Critical safety complaint
def test_4_critical_safety_complaint():
    req = PriorityRequest(
        complaint_text="Huge fire hazard with sparking transformer near fuel station.",
        domain="FIRE_DISASTER_EMERGENCY",
        subcategory="FIRE_HAZARD",
    )
    res = evaluate_priority(req)
    assert res.priority == "CRITICAL"
    assert res.override_applied is True
    assert res.sla_target_hours == 6


# 5. Garbage overflow with scale and duration
def test_5_garbage_overflow_scale_and_duration():
    req = PriorityRequest(
        complaint_text="Massive garbage overflow piling up for 2 weeks in residential complex.",
        domain="SOLID_WASTE",
        subcategory="OVERFLOWING_GARBAGE_BIN",
        affected_count=150,
        duration_days=14.0,
    )
    res = evaluate_priority(req)
    assert res.factors.affected_population >= 0.70
    assert res.factors.duration >= 0.75
    assert res.priority in ("HIGH", "CRITICAL")


# 6. Water main leak with infrastructure criticality
def test_6_water_main_leak_infrastructure():
    req = PriorityRequest(
        complaint_text="Water main leak pipeline burst flooding avenue.",
        domain="WATER_SUPPLY",
        subcategory="WATER_MAIN_LEAK",
        category_code="WTR-LEAK",
        affected_count=300,
    )
    res = evaluate_priority(req)
    assert res.factors.infrastructure_criticality >= 0.85
    assert res.factors.affected_population >= 0.90
    assert res.priority in ("HIGH", "CRITICAL")


# 7. Category-aware baseline test
def test_7_category_aware_baseline():
    req_walk = PriorityRequest(
        complaint_text="Paver block loose on quiet sidewalk.",
        category_code="RDW-WALK",
    )
    res_walk = evaluate_priority(req_walk)

    req_leak = PriorityRequest(
        complaint_text="Water pipeline burst leaking heavily.",
        category_code="WTR-LEAK",
    )
    res_leak = evaluate_priority(req_leak)

    assert res_leak.priority_score > res_walk.priority_score


# 8. Electrical hazard live wire test
def test_8_electrical_hazard_live_wire():
    req = PriorityRequest(
        complaint_text="Exposed live wire hanging near pedestrian walkway.",
        domain="STREET_LIGHTING_ELECTRICAL",
        subcategory="EXPOSED_ELECTRICAL_WIRE",
    )
    res = evaluate_priority(req)
    assert res.priority == "CRITICAL"
    assert res.override_applied is True
    assert "Safety Override" in res.risk_factors[0]


# 9. Repeated complaints (recurrence contribution)
def test_9_repeated_complaints_recurrence():
    req_single = PriorityRequest(
        complaint_text="Drainage block on street.",
        domain="SEWERAGE_DRAINAGE",
        recurrence_count=0,
    )
    res_single = evaluate_priority(req_single)

    req_repeat = PriorityRequest(
        complaint_text="Drainage block on street.",
        domain="SEWERAGE_DRAINAGE",
        recurrence_count=5,
    )
    res_repeat = evaluate_priority(req_repeat)

    assert res_repeat.factors.recurrence > res_single.factors.recurrence
    assert res_repeat.priority_score > res_single.priority_score


# 10. Long-duration complaint contribution
def test_10_long_duration_complaint():
    req_new = PriorityRequest(
        complaint_text="Streetlight flickering.",
        duration_days=0.5,
    )
    res_new = evaluate_priority(req_new)

    req_old = PriorityRequest(
        complaint_text="Streetlight flickering.",
        duration_days=20.0,
    )
    res_old = evaluate_priority(req_old)

    assert res_old.factors.duration > res_new.factors.duration
    assert res_old.priority_score > res_new.priority_score


# 11. SLA near breach contribution
def test_11_sla_near_breach():
    req = PriorityRequest(
        complaint_text="Pothole on main avenue.",
        sla_info=SLAData(
            sla_duration_hours=24,
            remaining_sla_hours=2.0,
            is_near_breach=True,
        ),
    )
    res = evaluate_priority(req)
    assert res.factors.sla_risk >= 0.85
    assert any("SLA" in rf for rf in res.risk_factors)


# 12. Hotspot data contribution
def test_12_hotspot_data_contribution():
    req = PriorityRequest(
        complaint_text="Overflowing waste bin.",
        hotspot_info=HotspotData(
            is_hotspot=True,
            complaint_density=0.85,
            cluster_count=8,
        ),
    )
    res = evaluate_priority(req)
    assert res.factors.hotspot >= 0.85
    assert any("Hotspot" in rf for rf in res.risk_factors)


# 13. Risk engine data contribution
def test_13_risk_engine_contribution():
    req = PriorityRequest(
        complaint_text="Water contamination query.",
        risk_info=RiskData(
            risk_score=0.90,
            risk_level="HIGH",
            safety_risk=0.85,
        ),
    )
    res = evaluate_priority(req)
    assert res.factors.risk == 0.90
    assert any("Risk Engine" in rf for rf in res.risk_factors)


# 14. Safety override flag and reason
def test_14_safety_override_flag_and_reason():
    req = PriorityRequest(
        complaint_text="Building collapse risk with visible wall cracks.",
        domain="BUILDING_CONSTRUCTION",
        subcategory="STRUCTURAL_DAMAGE",
    )
    res = evaluate_priority(req)
    assert res.priority == "CRITICAL"
    assert res.override_applied is True
    assert res.override_reason is not None
    assert "Building Structure Collapse Risk" in res.override_reason


# 15. Graceful degradation on missing hotspot/risk optional data
def test_15_missing_optional_data_graceful():
    req = PriorityRequest(
        complaint_text="Streetlight outage on corner.",
        domain="STREET_LIGHTING_ELECTRICAL",
        hotspot_info=None,
        risk_info=None,
        sla_info=None,
    )
    res = evaluate_priority(req)
    assert res.factors.hotspot == 0.0
    assert res.factors.risk == 0.0
    assert res.status in ("SUCCESS", "LOW_CONFIDENCE")


# 16. Missing DB category mapping handling
def test_16_missing_category_mapping():
    req = PriorityRequest(
        complaint_text="Custom civic complaint with no DB category code.",
        category_code=None,
    )
    res = evaluate_priority(req)
    assert 0.0 <= res.priority_score <= 1.0
    assert res.priority in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


# 17. Empty / short complaint description (low confidence)
def test_17_short_description_low_confidence():
    req = PriorityRequest(
        complaint_text="Bad",
    )
    res = evaluate_priority(req)
    assert res.confidence < 0.60
    assert len(res.warnings) > 0


# 18. Threshold boundary behavior test
def test_18_threshold_boundary_behavior():
    req_low = PriorityRequest(
        complaint_text="Bench painting.",
        domain="GARDENS_GREENERY",
    )
    res_low = evaluate_priority(req_low)
    assert res_low.priority_score < PRIORITY_THRESHOLDS["low_max"]

    req_crit = PriorityRequest(
        complaint_text="Live wire electrocution fire hazard emergency.",
        domain="FIRE_DISASTER_EMERGENCY",
    )
    res_crit = evaluate_priority(req_crit)
    assert res_crit.priority_score >= PRIORITY_THRESHOLDS["high_max"]


# 19. HTTP API Endpoint POST /api/v1/ai/priority
def test_19_priority_endpoint_http():
    payload = {
        "complaint_text": "Live wire sparking outside elementary school.",
        "domain": "FIRE_DISASTER_EMERGENCY",
        "subcategory": "EXPOSED_ELECTRICAL_WIRE",
        "affected_count": 50,
    }
    response = client.post("/api/v1/ai/priority", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["priority"] == "CRITICAL"
    assert data["priority_score"] >= 0.85
    assert "factors" in data
    assert data["factors"]["severity"] >= 0.50
    assert data["override_applied"] is True
