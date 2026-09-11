"""
CivicFlow AI - Advanced Escalation Agent
===========================================

Evaluates active tickets against multi-level escalation matrices, resolves candidate officer
reassignments, computes SLA breach penalty points, generates target notification drafts,
and outputs complete audit action logs.
"""

from typing import List, Optional, Tuple

from app.config.escalation_config import (
    DEPARTMENT_HEAD_CONTACTS,
    ESCALATION_TIER_THRESHOLDS,
    PRIORITY_PENALTY_WEIGHTS,
)
from app.models.escalation import (
    EscalationRequest,
    EscalationResult,
    OfficerCandidate,
)

DEFAULT_DEPARTMENT_CANDIDATES = {
    "SAN": [
        OfficerCandidate(officer_id="OFF-SAN-02", name="Officer Marcus Vance", active_tickets=1, skills=["SOLID_WASTE", "RECYCLING"]),
        OfficerCandidate(officer_id="OFF-SAN-05", name="Officer Elena Rostova", active_tickets=3, skills=["HAZMAT", "SANITATION"]),
    ],
    "RDW": [
        OfficerCandidate(officer_id="OFF-RDW-04", name="Engineer Carlos Mendez", active_tickets=1, skills=["PAVEMENT", "BRIDGES"]),
        OfficerCandidate(officer_id="OFF-RDW-07", name="Engineer Priya Sharma", active_tickets=2, skills=["DRAINAGE", "ROADWORKS"]),
    ],
    "WTR": [
        OfficerCandidate(officer_id="OFF-WTR-03", name="Technician David Kim", active_tickets=0, skills=["WATER_MAIN", "LEAKS"]),
        OfficerCandidate(officer_id="OFF-WTR-06", name="Technician Sarah Jenkins", active_tickets=2, skills=["SEWERAGE", "VALVES"]),
    ],
    "ELE": [
        OfficerCandidate(officer_id="OFF-ELE-02", name="Electrician Robert Chen", active_tickets=1, skills=["HIGH_VOLTAGE", "STREETLIGHTS"]),
        OfficerCandidate(officer_id="OFF-ELE-08", name="Electrician Fatima Al-Mansoor", active_tickets=2, skills=["TRANSFORMERS", "GRID"]),
    ],
    "PRK": [
        OfficerCandidate(officer_id="OFF-PRK-03", name="Arborist Daniel Miller", active_tickets=0, skills=["TREES", "PARKS"]),
        OfficerCandidate(officer_id="OFF-PRK-05", name="Supervisor Chloe Bennett", active_tickets=1, skills=["LANDSCAPING"]),
    ],
}


def _select_reassignment_candidate(
    department_code: str,
    candidates: Optional[List[OfficerCandidate]],
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    pool = candidates
    if not pool:
        dept_upper = (department_code or "SAN").upper()
        pool = DEFAULT_DEPARTMENT_CANDIDATES.get(dept_upper, DEFAULT_DEPARTMENT_CANDIDATES["SAN"])

    available = [c for c in pool if c.is_available]
    if not available:
        return None, None, None

    sorted_candidates = sorted(available, key=lambda c: c.active_tickets)
    best = sorted_candidates[0]
    return best.officer_id, best.name, best.active_tickets


def evaluate_escalation(request: EscalationRequest) -> EscalationResult:
    """Evaluates ticket priority, SLA status, and risk metrics to trigger multi-stage escalation."""
    audit_trace: List[str] = []
    warnings: List[str] = []
    dept_code = (request.department_code or "SAN").upper()
    priority_upper = (request.priority or "MEDIUM").upper()

    audit_trace.append(f"Initiated Escalation Engine evaluation for ticket '{request.ticket_id}'.")
    audit_trace.append(f"Input Context: Dept={dept_code}, Priority={priority_upper}, SLA Status={request.sla_status}, SLA Ratio={request.sla_ratio:.2f}, Risk Prob={request.breach_probability:.2f}.")

    dept_contact_info = DEPARTMENT_HEAD_CONTACTS.get(dept_code, DEPARTMENT_HEAD_CONTACTS["SAN"])
    affected = request.affected_count or 0

    # -----------------------------------------------------------------------
    # 1. Multi-Stage Escalation Tier Determination
    # -----------------------------------------------------------------------
    is_emergency_breach = request.is_breached and (request.is_safety_override or affected >= ESCALATION_TIER_THRESHOLDS["tier_4_affected_threshold"])

    if is_emergency_breach:
        escalation_tier = "TIER_4_EMERGENCY_OVERRIDE"
        action_required = "EMERGENCY_DISPATCH_TRIGGERED"
        target_role = "EMERGENCY_RESPONSE_UNIT"
        target_contact = f"EMERGENCY-DISPATCH-{dept_code} | Hotline 911-CIVIC"
        audit_trace.append("CRITICAL TRIGGER: Emergency breach condition met (Safety Override or High Citizen Impact). Tier 4 active.")

    elif request.is_breached or request.sla_ratio >= ESCALATION_TIER_THRESHOLDS["tier_3_ratio"] or request.sla_status == "BREACHED":
        escalation_tier = "TIER_3_DEPARTMENT_HEAD_ESCALATION"
        action_required = "EXEC_BOARD_ESCALATED"
        target_role = "DEPARTMENT_HEAD"
        target_contact = f"{dept_contact_info['title']} ({dept_contact_info['email']})"
        audit_trace.append(f"BREACH TRIGGER: SLA deadline exceeded. Escalated to Department Head: {dept_contact_info['title']}.")

    elif request.sla_ratio >= ESCALATION_TIER_THRESHOLDS["tier_2_ratio"] or request.breach_probability >= ESCALATION_TIER_THRESHOLDS["tier_2_prob"] or request.sla_tier == "CRITICAL_RISK":
        escalation_tier = "TIER_2_PRE_BREACH_REASSIGNMENT"
        action_required = "REASSIGNMENT_RECOMMENDED"
        target_role = "REASSIGNED_OFFICER"
        target_contact = f"Dept {dept_code} Dispatch Coordinator"
        audit_trace.append("PRE-BREACH TRIGGER: SLA ratio >= 90% or breach probability >= 85%. Triggering automated reassignment recommendation.")

    elif request.sla_ratio >= ESCALATION_TIER_THRESHOLDS["tier_1_ratio"] or request.breach_probability >= ESCALATION_TIER_THRESHOLDS["tier_1_prob"] or request.sla_tier == "NEAR_BREACH":
        escalation_tier = "TIER_1_SUPERVISOR_ALERT"
        action_required = "SUPERVISOR_NOTIFIED"
        target_role = "SUPERVISOR"
        target_contact = f"Supervisor Zone-{dept_code} (sup.{dept_code.lower()}@civicflow.gov)"
        audit_trace.append("WARNING TRIGGER: SLA ratio >= 75% or breach probability >= 70%. Notifying assigned officer & zone supervisor.")

    else:
        escalation_tier = "TIER_0_NOMINAL"
        action_required = "NONE"
        target_role = "ASSIGNED_OFFICER"
        target_contact = request.current_officer_id or "OFF-ASSIGNED-01"
        audit_trace.append("NOMINAL STATUS: Ticket is operating within normal SLA parameters. No escalation action needed.")

    # -----------------------------------------------------------------------
    # 2. Officer Reassignment Selection (Tiers 2, 3, 4)
    # -----------------------------------------------------------------------
    rec_officer_id, rec_officer_name, rec_active_tickets = _select_reassignment_candidate(dept_code, request.candidate_officers)
    if escalation_tier in {"TIER_2_PRE_BREACH_REASSIGNMENT", "TIER_3_DEPARTMENT_HEAD_ESCALATION", "TIER_4_EMERGENCY_OVERRIDE"}:
        if rec_officer_id:
            audit_trace.append(f"Reassignment Engine selected '{rec_officer_name}' ({rec_officer_id}) with {rec_active_tickets} active tickets.")
        else:
            warnings.append("No available fallback officer found for automated reassignment.")

    # -----------------------------------------------------------------------
    # 3. Financial & Compliance SLA Penalty Calculation
    # -----------------------------------------------------------------------
    penalty_score = 0.0
    if request.is_breached or request.sla_ratio > 1.0:
        overtime_ratio = max(0.0, request.sla_ratio - 1.0)
        base_rate = PRIORITY_PENALTY_WEIGHTS.get(priority_upper, 20.0)
        penalty_score = round((1.0 + overtime_ratio * 2.0) * base_rate, 2)
        if request.is_safety_override:
            penalty_score = round(penalty_score * 1.5, 2)
        audit_trace.append(f"Penalty calculation: SLA breach incurred compliance penalty score of {penalty_score} points.")

    # -----------------------------------------------------------------------
    # 4. Notification Draft Generator
    # -----------------------------------------------------------------------
    if escalation_tier == "TIER_4_EMERGENCY_OVERRIDE":
        notification_draft = (
            f"[🚨 EMERGENCY DISPATCH ALERT]\n"
            f"Ticket ID: {request.ticket_id}\n"
            f"Department: {dept_code} | Priority: CRITICAL\n"
            f"Hazard Context: {request.complaint_text or 'Critical Safety Hazard Breached'}\n"
            f"Affected Citizens: {affected} | SLA Ratio: {request.sla_ratio * 100:.0f}%\n"
            f"DIRECTIVE: Deploy emergency response unit immediately. Cross-department coordination activated."
        )
    elif escalation_tier == "TIER_3_DEPARTMENT_HEAD_ESCALATION":
        notification_draft = (
            f"[💥 DEPARTMENT HEAD ESCALATION NOTICE]\n"
            f"To: {dept_contact_info['title']}\n"
            f"Ticket ID: {request.ticket_id} has BREACHED allocated SLA duration.\n"
            f"Priority: {priority_upper} | Overtime Penalty: {penalty_score} pts\n"
            f"Recommended Reassignment: {rec_officer_name or 'Unassigned'} ({rec_officer_id or 'N/A'})\n"
            f"ACTION REQUIRED: Review ticket bottleneck and approve priority resource dispatch."
        )
    elif escalation_tier == "TIER_2_PRE_BREACH_REASSIGNMENT":
        notification_draft = (
            f"[⚠️ PRE-BREACH REASSIGNMENT DIRECTIVE]\n"
            f"Ticket ID: {request.ticket_id} is at {request.sla_ratio * 100:.0f}% SLA capacity (Risk Prob: {request.breach_probability * 100:.0f}%).\n"
            f"Currently Assigned: {request.current_officer_id}\n"
            f"Recommended Fallback Officer: {rec_officer_name} ({rec_officer_id}) [Load: {rec_active_tickets} active tickets]\n"
            f"ACTION REQUIRED: Reassign ticket immediately to prevent SLA breach."
        )
    elif escalation_tier == "TIER_1_SUPERVISOR_ALERT":
        notification_draft = (
            f"[⚡ SLA WARNING NOTIFICATION]\n"
            f"Ticket ID: {request.ticket_id} has reached 75%+ SLA threshold.\n"
            f"Assigned Officer: {request.current_officer_id} | Priority: {priority_upper}\n"
            f"ACTION REQUIRED: Monitor resolution progress. Supervisor check-in recommended."
        )
    else:
        notification_draft = (
            f"[ℹ️ SLA STATUS NORMAL]\n"
            f"Ticket ID: {request.ticket_id} is operating on track within standard SLA targets."
        )

    # -----------------------------------------------------------------------
    # 5. Explanation Synthesis
    # -----------------------------------------------------------------------
    explanation = (
        f"Escalation evaluated to '{escalation_tier}' with action '{action_required}'. "
        f"Target recipient: {target_role} ({target_contact}). "
        f"Penalty points: {penalty_score}. "
        f"Reassignment candidate: {rec_officer_name or 'None'} ({rec_officer_id or 'None'})."
    )

    audit_trace.append("Escalation evaluation cycle successfully finalized.")

    return EscalationResult(
        status="SUCCESS",
        ticket_id=request.ticket_id,
        escalation_tier=escalation_tier,
        action_required=action_required,
        target_role=target_role,
        target_contact=target_contact,
        recommended_officer_id=rec_officer_id,
        recommended_officer_name=rec_officer_name,
        penalty_score=penalty_score,
        notification_draft=notification_draft,
        audit_trace=audit_trace,
        explanation=explanation,
        warnings=warnings,
    )
