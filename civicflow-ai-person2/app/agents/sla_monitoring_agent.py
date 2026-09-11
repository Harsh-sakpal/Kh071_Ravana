"""
CivicFlow AI - Advanced SLA Monitoring Agent
==============================================

Evaluates active municipal tickets against SLA target resolution deadlines, incorporating
category SLA rules, location urgency discounts, 6-factor breach probability modeling,
multi-stage risk alert tiers, and factor score breakdowns.
"""

from datetime import datetime, timezone
from typing import List

from app.config.priority_config import SLA_TARGET_HOURS
from app.config.sla_config import (
    CATEGORY_SLA_DURATIONS,
    SLA_FEATURE_WEIGHTS,
    SLA_TIER_THRESHOLDS,
    SUBCATEGORY_COMPLEXITY,
)
from app.models.sla_monitoring import SLAFactors, SLAMonitorRequest, SLAMonitorResult

EXEMPT_STATUSES = {"RESOLVED", "CLOSED"}


def _ensure_timezone(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _compute_officer_workload_factor(workload: int | None) -> float:
    if workload is None:
        return 0.20
    if workload <= 1:
        return 0.10
    if workload <= 4:
        return 0.40
    if workload <= 8:
        return 0.70
    return 1.00


def _compute_queue_factor(queue_count: int | None) -> float:
    if queue_count is None:
        return 0.15
    if queue_count <= 5:
        return 0.10
    if queue_count <= 20:
        return 0.45
    if queue_count <= 50:
        return 0.75
    return 1.00


def evaluate_sla_status(request: SLAMonitorRequest) -> SLAMonitorResult:
    """Evaluates ticket SLA tracking, multi-tier risk status, 6-factor breach probability, and alerts."""
    warnings: List[str] = []

    # 1. Resolve Target SLA Hours (Category Rules -> Priority Defaults)
    priority_upper = (request.priority or "MEDIUM").upper()
    sla_target_hours = request.sla_target_hours

    if not sla_target_hours:
        if request.category_code and request.category_code in CATEGORY_SLA_DURATIONS:
            sla_target_hours = CATEGORY_SLA_DURATIONS[request.category_code]
        else:
            sla_target_hours = SLA_TARGET_HOURS.get(priority_upper, 48)

    # Apply sensitive location SLA reduction (25% reduction for schools/hospitals)
    if request.is_sensitive_location:
        sla_target_hours = max(4, int(round(sla_target_hours * 0.75)))
        warnings.append("Sensitive location discount applied: SLA duration reduced by 25%.")

    # 2. Time Calculations
    now_dt = _ensure_timezone(request.current_time or datetime.now(timezone.utc))
    created_dt = _ensure_timezone(request.created_at or now_dt)

    if created_dt > now_dt:
        warnings.append("Ticket creation timestamp is in the future. Resetting created_at to evaluation time.")
        created_dt = now_dt

    elapsed_seconds = max(0.0, (now_dt - created_dt).total_seconds())
    elapsed_hours = round(elapsed_seconds / 3600.0, 2)
    status_upper = (request.status or "ASSIGNED").upper()

    # 3. Handle Exempt Statuses (RESOLVED, CLOSED)
    if status_upper in EXEMPT_STATUSES:
        sla_ratio = round(elapsed_hours / sla_target_hours, 4)
        factors = SLAFactors(
            time_elapsed=0.0,
            officer_workload=0.0,
            department_queue=0.0,
            complexity=0.0,
            location_urgency=0.0,
            historical_breach=0.0,
        )
        explanation = f"Ticket is in '{status_upper}' status. SLA tracking is EXEMPT."
        return SLAMonitorResult(
            status="SUCCESS",
            ticket_id=request.ticket_id,
            sla_status="EXEMPT",
            sla_tier="EXEMPT",
            elapsed_hours=elapsed_hours,
            remaining_hours=0.0,
            sla_target_hours=sla_target_hours,
            sla_ratio=sla_ratio,
            is_warning_triggered=False,
            is_breached=False,
            breach_probability=0.0,
            sla_factors=factors,
            explanation=explanation,
            recommendation_action="TICKET_RESOLVED_NO_ACTION_REQUIRED",
            warnings=warnings,
        )

    # 4. Handle Active Ticket Statuses
    remaining_hours = max(0.0, round(sla_target_hours - elapsed_hours, 2))
    sla_ratio = round(elapsed_hours / sla_target_hours, 4)

    # 5. Extract 6 Normalized Feature Factors
    f_time = min(1.0, sla_ratio)
    f_workload = _compute_officer_workload_factor(request.officer_workload_count)
    f_queue = _compute_queue_factor(request.department_queue_count)
    f_complexity = SUBCATEGORY_COMPLEXITY.get(request.subcategory, 0.40)
    f_location = 0.85 if request.is_sensitive_location else 0.20
    f_historical = min(1.0, max(0.0, request.historical_breach_rate)) if request.historical_breach_rate is not None else 0.15

    factors = SLAFactors(
        time_elapsed=round(f_time, 2),
        officer_workload=round(f_workload, 2),
        department_queue=round(f_queue, 2),
        complexity=round(f_complexity, 2),
        location_urgency=round(f_location, 2),
        historical_breach=round(f_historical, 2),
    )

    # 6. Weighted 6-Factor Continuous Breach Probability Calculation
    weighted_prob = (
        f_time * SLA_FEATURE_WEIGHTS["time_elapsed"]
        + f_workload * SLA_FEATURE_WEIGHTS["officer_workload"]
        + f_queue * SLA_FEATURE_WEIGHTS["department_queue"]
        + f_complexity * SLA_FEATURE_WEIGHTS["complexity"]
        + f_location * SLA_FEATURE_WEIGHTS["location_urgency"]
        + f_historical * SLA_FEATURE_WEIGHTS["historical_breach"]
    )

    # 7. Evaluate Multi-Stage Alert Tier & Status
    if elapsed_hours >= sla_target_hours:
        sla_status = "BREACHED"
        sla_tier = "BREACHED"
        is_breached = True
        is_warning_triggered = True
        breach_probability = 1.00
        recommendation_action = "IMMEDIATE_ESCALATION_REQUIRED"
    elif sla_ratio >= SLA_TIER_THRESHOLDS["critical_risk"]:
        sla_status = "NEAR_BREACH"
        sla_tier = "CRITICAL_RISK"
        is_breached = False
        is_warning_triggered = True
        breach_probability = min(0.99, max(0.85, round(weighted_prob, 4)))
        recommendation_action = "PREPARE_PRE_BREACH_ESCALATION"
    elif sla_ratio >= SLA_TIER_THRESHOLDS["near_breach"]:
        sla_status = "NEAR_BREACH"
        sla_tier = "NEAR_BREACH"
        is_breached = False
        is_warning_triggered = True
        breach_probability = min(0.89, max(0.70, round(weighted_prob, 4)))
        recommendation_action = "TRIGGER_NEAR_BREACH_WARNING"
    elif sla_ratio >= SLA_TIER_THRESHOLDS["progressing"]:
        sla_status = "WITHIN_SLA"
        sla_tier = "PROGRESSING"
        is_breached = False
        is_warning_triggered = False
        breach_probability = min(0.65, max(0.35, round(weighted_prob, 4)))
        recommendation_action = "MONITOR_PROGRESS"
    else:
        sla_status = "WITHIN_SLA"
        sla_tier = "ON_TRACK"
        is_breached = False
        is_warning_triggered = False
        breach_probability = min(0.40, max(0.0, round(weighted_prob, 4)))
        recommendation_action = "MAINTAIN_NORMAL_MONITORING"

    # 8. Dynamic Explanation Generation
    explanation_parts = [
        f"SLA status is '{sla_status}' (Tier: {sla_tier}).",
        f"Elapsed time: {elapsed_hours:.1f}h / {sla_target_hours}h target ({sla_ratio * 100:.0f}% SLA consumed, {remaining_hours:.1f}h remaining).",
        f"Breach probability: {breach_probability * 100:.0f}% based on time elapsed ({f_time:.2f}) and officer workload ({f_workload:.2f}).",
    ]

    explanation = " ".join(explanation_parts)

    return SLAMonitorResult(
        status="SUCCESS",
        ticket_id=request.ticket_id,
        sla_status=sla_status,
        sla_tier=sla_tier,
        elapsed_hours=elapsed_hours,
        remaining_hours=remaining_hours,
        sla_target_hours=sla_target_hours,
        sla_ratio=sla_ratio,
        is_warning_triggered=is_warning_triggered,
        is_breached=is_breached,
        breach_probability=breach_probability,
        sla_factors=factors,
        explanation=explanation,
        recommendation_action=recommendation_action,
        warnings=warnings,
    )
