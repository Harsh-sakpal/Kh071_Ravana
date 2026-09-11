"""
CivicFlow AI - Priority Agent (DIPEX-Style Continuous Scoring System)
================================================----------------======

Calculates a continuous priority score from 0.0 to 1.0 across 10 normalized
feature groups inspired by DIPEX municipal complaint prioritization principles:
    1. severity_score
    2. public_impact_score
    3. urgency_score
    4. infrastructure_criticality_score
    5. affected_population_score
    6. recurrence_score
    7. duration_score
    8. sla_risk_score
    9. hotspot_score
    10. risk_score

Features are weighted using configurable settings in `app.config.priority_config`.
The continuous score is converted to LOW / MEDIUM / HIGH / CRITICAL priority levels
via configurable thresholds, with deterministic safety overrides for immediate
life/property hazards.
"""

import re
from typing import List, Tuple

from app.config.priority_config import (
    CATEGORY_CODE_BASELINES,
    CRITICAL_INFRASTRUCTURE_PATTERNS,
    FEATURE_WEIGHTS,
    PRIORITY_THRESHOLDS,
    SAFETY_OVERRIDE_PATTERNS,
    SLA_TARGET_HOURS,
)
from app.models.priority import PriorityFactors, PriorityRequest, PriorityResult

CRITICAL_SUBCATEGORIES = {
    "OPEN_MANHOLE",
    "MISSING_MANHOLE",
    "EXPOSED_ELECTRICAL_WIRE",
    "ELECTRICAL_HAZARD",
    "SPARKING_EQUIPMENT",
    "WATER_MAIN_LEAK",
    "NO_WATER_SUPPLY",
    "CONTAMINATED_WATER",
    "WATER_CONTAMINATION",
    "DANGEROUS_BUILDING",
    "STRUCTURAL_DAMAGE",
    "COLLAPSE_RISK",
    "FIRE_HAZARD",
    "CHEMICAL_CONTAMINATION",
    "ILLEGAL_CHEMICAL_DUMPING",
    "DEAD_ANIMAL_CARCASS",
    "DEAD_ANIMAL_WASTE",
}

HIGH_SUBCATEGORIES = {
    "GARBAGE_NOT_COLLECTED",
    "OVERFLOWING_GARBAGE_BIN",
    "ILLEGAL_DUMPING",
    "MOSQUITO_BREEDING",
    "STRAY_ANIMAL_ISSUE",
    "WATERLOGGING",
    "FLOODED_ROAD",
    "BLOCKED_STORM_DRAIN",
    "POTHOLE",
    "LARGE_POTHOLE",
    "FAULTY_TRAFFIC_SIGNAL",
    "TRAFFIC_SIGNAL_ISSUE",
    "STREETLIGHT_OUTAGE",
    "DARK_STREET",
    "FALLEN_TREE",
    "TREE_BRANCH_HAZARD",
}

SENSITIVE_LOCATION_PATTERNS = [
    r"\bschool\b",
    r"\bhospital\b",
    r"\bclinic\b",
    r"\bdaycare\b",
    r"\bnursery\b",
    r"\bplayground\b",
    r"\bpedestrian walkway\b",
    r"\bnursing home\b",
    r"\belderly care\b",
    r"\bmarket\b",
    r"\bbusy road\b",
    r"\bmain avenue\b",
]

URGENCY_KEYWORD_WEIGHTS = [
    (r"\bemergency\b", 0.90),
    (r"\bimmediate danger\b", 0.95),
    (r"\blife threatening\b", 0.95),
    (r"\bcausing accidents\b", 0.85),
    (r"\bsevere risk\b", 0.85),
    (r"\bimmediately\b", 0.75),
    (r"\burgent\b", 0.75),
    (r"\bdangerous\b", 0.75),
    (r"\bcritical\b", 0.80),
    (r"\bsevere\b", 0.70),
    (r"\bmassive\b", 0.75),
    (r"\boverflow\b", 0.75),
    (r"\bflooding\b", 0.75),
    (r"\bburst\b", 0.80),
    (r"\bexposed wire\b", 0.90),
    (r"\bfire\b", 0.90),
    (r"\bcontamination\b", 0.85),
    (r"\bsparking\b", 0.85),
]


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compute_severity_score(request: PriorityRequest, text_norm: str) -> float:
    score = 0.20  # Base score for minor inconvenience

    # Check category baseline if available
    if request.category_code and request.category_code in CATEGORY_CODE_BASELINES:
        baseline = CATEGORY_CODE_BASELINES[request.category_code]
        if baseline == "CRITICAL":
            score = max(score, 0.90)
        elif baseline == "HIGH":
            score = max(score, 0.70)
        elif baseline == "MEDIUM":
            score = max(score, 0.45)
        elif baseline == "LOW":
            score = max(score, 0.20)

    # Check subcategory lists
    if request.subcategory in CRITICAL_SUBCATEGORIES:
        score = max(score, 0.90)
    elif request.subcategory in HIGH_SUBCATEGORIES:
        score = max(score, 0.65)

    # Check domain defaults
    if request.domain in ("FIRE_DISASTER_EMERGENCY", "STORM_WATER", "SEWERAGE_DRAINAGE"):
        score = max(score, 0.65)
    elif request.domain in ("SOLID_WASTE", "PUBLIC_HEALTH_SANITATION", "WATER_SUPPLY", "ROADS_FOOTPATHS"):
        score = max(score, 0.45)

    return min(1.0, score)


def _compute_public_impact_score(request: PriorityRequest, text_norm: str) -> float:
    score = 0.20
    # Location context exposure
    for pattern in SENSITIVE_LOCATION_PATTERNS:
        if re.search(pattern, text_norm):
            score = max(score, 0.75)
            break

    if request.location_context:
        loc_norm = _normalize_text(request.location_context)
        for pattern in SENSITIVE_LOCATION_PATTERNS:
            if re.search(pattern, loc_norm):
                score = max(score, 0.85)
                break

    return min(1.0, score)


def _compute_urgency_score(text_norm: str) -> float:
    score = 0.10
    matched_weights = []
    for pattern, weight in URGENCY_KEYWORD_WEIGHTS:
        if re.search(pattern, text_norm):
            matched_weights.append(weight)

    if matched_weights:
        # Highest match weight + minor boost for multiple urgency signals
        highest = max(matched_weights)
        boost = min(0.15, (len(matched_weights) - 1) * 0.05)
        score = highest + boost

    return min(1.0, score)


def _compute_infrastructure_criticality_score(request: PriorityRequest, text_norm: str) -> float:
    score = 0.15

    # Check critical infrastructure keywords
    for pattern in CRITICAL_INFRASTRUCTURE_PATTERNS:
        if re.search(pattern, text_norm):
            score = max(score, 0.85)
            break

    # Domain criticality
    if request.domain in ("WATER_SUPPLY", "STREET_LIGHTING_ELECTRICAL", "SEWERAGE_DRAINAGE"):
        score = max(score, 0.60)
    elif request.domain in ("ROADS_FOOTPATHS", "STORM_WATER", "FIRE_DISASTER_EMERGENCY"):
        score = max(score, 0.50)

    # Subcategory criticality
    if request.subcategory in ("WATER_MAIN_LEAK", "NO_WATER_SUPPLY", "EXPOSED_ELECTRICAL_WIRE", "SPARKING_EQUIPMENT"):
        score = max(score, 0.90)

    return min(1.0, score)


def _compute_affected_population_score(affected_count: int | None) -> float:
    if not affected_count or affected_count <= 1:
        return 0.10  # Individual
    if affected_count <= 10:
        return 0.30  # Household / Small building
    if affected_count <= 50:
        return 0.50  # Society / Street
    if affected_count <= 200:
        return 0.75  # Locality / Neighborhood
    return 0.95  # Large area / Multiple localities


def _compute_recurrence_score(request: PriorityRequest, text_norm: str) -> float:
    if request.recurrence_count is not None and request.recurrence_count > 0:
        if request.recurrence_count == 1:
            return 0.40
        if request.recurrence_count <= 3:
            return 0.70
        return 1.00

    # Fallback to text recurrence signals
    if re.search(r"\b(again|repeatedly|multiple times|every day|third time|recurring|frequent)\b", text_norm):
        return 0.60

    return 0.00


def _compute_duration_score(request: PriorityRequest, text_norm: str) -> float:
    if request.duration_days is not None and request.duration_days > 0:
        if request.duration_days <= 1.0:
            return 0.20
        if request.duration_days <= 3.0:
            return 0.45
        if request.duration_days <= 7.0:
            return 0.75
        return 1.00

    # Fallback text duration signals
    if re.search(r"\b(weeks|months|for days|since last week|several days)\b", text_norm):
        return 0.70

    return 0.20


def _compute_sla_risk_score(request: PriorityRequest) -> float:
    if not request.sla_info:
        # Default baseline SLA risk based on category baseline if available
        if request.category_code and request.category_code in CATEGORY_CODE_BASELINES:
            base = CATEGORY_CODE_BASELINES[request.category_code]
            if base == "CRITICAL":
                return 0.60
            if base == "HIGH":
                return 0.40
        return 0.20

    sla = request.sla_info
    if sla.is_breached:
        return 1.00
    if sla.is_near_breach:
        return 0.85

    if sla.remaining_sla_hours is not None and sla.sla_duration_hours and sla.sla_duration_hours > 0:
        ratio = sla.remaining_sla_hours / sla.sla_duration_hours
        if ratio <= 0.15:
            return 0.85
        if ratio <= 0.35:
            return 0.65
        if ratio <= 0.50:
            return 0.45

    return 0.20


def _compute_hotspot_score(request: PriorityRequest) -> float:
    if not request.hotspot_info:
        return 0.00

    hs = request.hotspot_info
    if not hs.is_hotspot:
        return 0.10

    score = 0.60
    if hs.complaint_density is not None:
        score = max(score, hs.complaint_density)
    if hs.cluster_count is not None and hs.cluster_count >= 5:
        score = max(score, 0.85)

    return min(1.0, score)


def _compute_risk_score(request: PriorityRequest) -> float:
    if not request.risk_info:
        return 0.00

    r = request.risk_info
    if r.risk_score is not None:
        return min(1.0, max(0.0, r.risk_score))

    sub_scores = [s for s in (r.safety_risk, r.health_risk, r.infrastructure_risk) if s is not None]
    if sub_scores:
        return min(1.0, max(sub_scores))

    if r.risk_level == "CRITICAL":
        return 0.95
    if r.risk_level == "HIGH":
        return 0.75
    if r.risk_level == "MEDIUM":
        return 0.50

    return 0.20


def evaluate_priority(request: PriorityRequest) -> PriorityResult:
    """Evaluates ticket priority, SLA, factor scores, safety overrides, and confidence."""
    warnings: List[str] = []
    risk_factors: List[str] = []

    text_norm = _normalize_text(request.complaint_text)
    if request.location_context:
        text_norm += " " + _normalize_text(request.location_context)

    # 1. Feature Score Extractions (10 DIPEX normalized feature groups)
    f_severity = _compute_severity_score(request, text_norm)
    f_impact = _compute_public_impact_score(request, text_norm)
    f_urgency = _compute_urgency_score(text_norm)
    f_infra = _compute_infrastructure_criticality_score(request, text_norm)
    f_pop = _compute_affected_population_score(request.affected_count)
    f_rec = _compute_recurrence_score(request, text_norm)
    f_dur = _compute_duration_score(request, text_norm)
    f_sla = _compute_sla_risk_score(request)
    f_hotspot = _compute_hotspot_score(request)
    f_risk = _compute_risk_score(request)

    factors = PriorityFactors(
        severity=round(f_severity, 2),
        public_impact=round(f_impact, 2),
        urgency=round(f_urgency, 2),
        infrastructure_criticality=round(f_infra, 2),
        affected_population=round(f_pop, 2),
        recurrence=round(f_rec, 2),
        duration=round(f_dur, 2),
        sla_risk=round(f_sla, 2),
        hotspot=round(f_hotspot, 2),
        risk=round(f_risk, 2),
    )

    # 2. Weighted Continuous Priority Score
    weighted_sum = (
        f_severity * FEATURE_WEIGHTS["severity"]
        + f_impact * FEATURE_WEIGHTS["public_impact"]
        + f_urgency * FEATURE_WEIGHTS["urgency"]
        + f_infra * FEATURE_WEIGHTS["infrastructure_criticality"]
        + f_pop * FEATURE_WEIGHTS["affected_population"]
        + f_rec * FEATURE_WEIGHTS["recurrence"]
        + f_dur * FEATURE_WEIGHTS["duration"]
        + f_sla * FEATURE_WEIGHTS["sla_risk"]
        + f_hotspot * FEATURE_WEIGHTS["hotspot"]
        + f_risk * FEATURE_WEIGHTS["risk"]
    )

    priority_score = min(1.0, max(0.0, round(weighted_sum, 4)))

    # 3. Priority Level Threshold Conversion
    if priority_score < PRIORITY_THRESHOLDS["low_max"]:
        assigned_priority = "LOW"
    elif priority_score < PRIORITY_THRESHOLDS["medium_max"]:
        assigned_priority = "MEDIUM"
    elif priority_score < PRIORITY_THRESHOLDS["high_max"]:
        assigned_priority = "HIGH"
    else:
        assigned_priority = "CRITICAL"

    # Risk factor tag collection
    if f_urgency >= 0.70:
        risk_factors.append("High Urgency Signals Detected")
    if f_impact >= 0.70:
        risk_factors.append("Sensitive Location / Public Space Exposure")
    if f_infra >= 0.70:
        risk_factors.append("Critical Infrastructure Exposure")
    if f_pop >= 0.70:
        risk_factors.append(f"High Population Impact (~{request.affected_count or '100+'} citizens)")
    if f_rec >= 0.60:
        risk_factors.append("Recurring Issue in Area")
    if f_dur >= 0.70:
        risk_factors.append("Prolonged Unresolved Issue")
    if f_sla >= 0.80:
        risk_factors.append("SLA Breach / Near-Breach Risk")
    if f_hotspot >= 0.60:
        risk_factors.append("High Hotspot Density Area")
    if f_risk >= 0.60:
        risk_factors.append("Risk Engine Alert Flagged")

    # 4. Deterministic Safety Override Check
    override_applied = False
    override_reason = None
    for pattern, desc in SAFETY_OVERRIDE_PATTERNS:
        if re.search(pattern, text_norm):
            override_applied = True
            override_reason = f"Deterministic Safety Override: Detected '{desc}'"
            assigned_priority = "CRITICAL"
            priority_score = max(priority_score, 0.95)
            risk_factors.insert(0, f"Safety Override: {desc}")
            break

    sla_hours = SLA_TARGET_HOURS[assigned_priority]

    # 5. Dynamic Explanation Generation
    explanation_parts = []
    if override_applied:
        explanation_parts.append(override_reason + ".")

    top_factors = []
    factor_dict = factors.model_dump()
    sorted_factors = sorted(factor_dict.items(), key=lambda x: x[1], reverse=True)
    for name, val in sorted_factors[:3]:
        if val >= 0.50:
            top_factors.append(f"{name.replace('_', ' ')} ({val:.2f})")

    if top_factors:
        explanation_parts.append(
            f"Assigned priority level '{assigned_priority}' with score {priority_score:.2f}. Key driving factors: {', '.join(top_factors)}."
        )
    else:
        explanation_parts.append(
            f"Assigned priority level '{assigned_priority}' with score {priority_score:.2f} based on standard operational metadata."
        )

    explanation = " ".join(explanation_parts)

    # 6. Confidence Score Calculation (Data quality & clarity)
    conf = 0.70
    if len(request.complaint_text.strip()) >= 40:
        conf += 0.10
    if request.domain or request.subcategory:
        conf += 0.10
    if request.category_code:
        conf += 0.05

    if len(request.complaint_text.strip()) < 15:
        conf -= 0.25
        warnings.append("Complaint description is very short, confidence reduced.")

    confidence = min(1.0, max(0.20, round(conf, 2)))
    status = "SUCCESS" if confidence >= 0.50 else "LOW_CONFIDENCE"

    return PriorityResult(
        status=status,
        complaint_id=request.complaint_id,
        priority=assigned_priority,
        priority_score=priority_score,
        sla_target_hours=sla_hours,
        factors=factors,
        override_applied=override_applied,
        override_reason=override_reason,
        confidence=confidence,
        explanation=explanation,
        reason=explanation,
        risk_factors=risk_factors,
        warnings=warnings,
    )
