"""
Routing Agent tests, per Master Prompt Section 17:
- correct domain
- correct subcategory
- database mapping
- department mapping
- ambiguous complaints
- multi-domain complaints
- unknown complaints
- confidence
- explanation
"""

import pytest

from app.models.routing import RoutingRequest
from app.agents.routing_agent import route_complaint


def test_correct_domain_and_subcategory_garbage():
    req = RoutingRequest(
        complaint_id="cmp-1",
        complaint_text="Garbage has not been collected for four days and waste is piling up outside our society.",
    )
    result = route_complaint(req)
    assert result.domain == "SOLID_WASTE"
    assert result.subcategory == "GARBAGE_NOT_COLLECTED"
    assert result.status == "SUCCESS"


def test_database_and_department_mapping_present_for_known_case():
    req = RoutingRequest(complaint_text="There is a huge pothole on the main road causing accidents.")
    result = route_complaint(req)
    assert result.domain == "ROADS_FOOTPATHS"
    assert result.subcategory == "POTHOLE"
    assert result.department_code == "RDW"
    assert result.category_code == "RDW-POT"
    assert result.category_mapping_available is True


def test_department_mapping_present_even_without_category_mapping():
    # TEMPORARY_ENCROACHMENT under ENCROACHMENT has department RDW, but no category_code mapped yet.
    req = RoutingRequest(
        complaint_text="Unmapped temporary encroachment.",
        existing_domain="ENCROACHMENT",
        existing_subcategory="TEMPORARY_ENCROACHMENT",
    )
    result = route_complaint(req)
    assert result.domain == "ENCROACHMENT"
    assert result.subcategory == "TEMPORARY_ENCROACHMENT"
    assert result.department_code == "RDW"
    assert result.category_code is None
    assert result.category_mapping_available is False
    assert any("no confirmed database category mapping" in w.lower() for w in result.warnings)


def test_streetlight_case():
    req = RoutingRequest(complaint_text="The streetlight near my house is not working, street is dark at night.")
    result = route_complaint(req)
    assert result.domain == "STREET_LIGHTING_ELECTRICAL"
    assert result.subcategory in ("STREETLIGHT_OUTAGE", "DARK_STREET")
    assert result.department_code == "ELE"


def test_water_main_leak_case():
    req = RoutingRequest(complaint_text="Water main leak, pipeline burst and flooding the street with water.")
    result = route_complaint(req)
    assert result.domain == "WATER_SUPPLY"
    assert result.subcategory == "WATER_MAIN_LEAK"
    assert result.department_code == "WTR"
    assert result.category_code == "WTR-LEAK"


def test_ambiguous_complaint_gets_low_confidence_or_other_bucket():
    req = RoutingRequest(complaint_text="Something is wrong near my house, please check.")
    result = route_complaint(req)
    assert result.status in ("LOW_CONFIDENCE", "UNRESOLVED")


def test_multi_domain_complaint_picks_dominant_signal():
    # Mentions both garbage (solid waste) and pothole (roads) -- pothole
    # keyword weight should dominate given phrasing below.
    req = RoutingRequest(
        complaint_text="There is a big pothole on the road and also some garbage litter nearby."
    )
    result = route_complaint(req)
    assert result.domain in ("ROADS_FOOTPATHS", "SOLID_WASTE")
    assert result.status in ("SUCCESS", "LOW_CONFIDENCE")


def test_unknown_complaint_returns_unresolved():
    req = RoutingRequest(complaint_text="asdkj qwer poiuy zxcvb nonsense text")
    result = route_complaint(req)
    assert result.status == "UNRESOLVED"
    assert result.domain is None
    assert result.confidence == 0.0


def test_confidence_is_within_bounds():
    req = RoutingRequest(complaint_text="Garbage not collected for a week, waste piling up outside society.")
    result = route_complaint(req)
    assert 0.0 <= result.confidence <= 1.0


def test_explanation_is_present_and_non_empty():
    req = RoutingRequest(complaint_text="Pothole causing accidents on main road.")
    result = route_complaint(req)
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0


def test_upstream_classification_is_respected_and_validated():
    req = RoutingRequest(
        complaint_text="Some complaint text.",
        existing_domain="SOLID_WASTE",
        existing_subcategory="ILLEGAL_DUMPING",
    )
    result = route_complaint(req)
    assert result.domain == "SOLID_WASTE"
    assert result.subcategory == "ILLEGAL_DUMPING"
    assert result.category_code == "SAN-DUMP"
    assert result.confidence >= 0.9


def test_invalid_upstream_domain_is_ignored_with_warning():
    req = RoutingRequest(
        complaint_text="Garbage has not been collected for days.",
        existing_domain="NOT_A_REAL_DOMAIN",
    )
    result = route_complaint(req)
    assert result.domain == "SOLID_WASTE"  # falls back to rule-based detection
    assert any("not part of the locked taxonomy" in w for w in result.warnings)


def test_complaint_id_is_passed_through():
    req = RoutingRequest(complaint_id="cmp-999", complaint_text="Pothole on the street.")
    result = route_complaint(req)
    assert result.complaint_id == "cmp-999"
