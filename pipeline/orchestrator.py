"""
CivicFlow AI — End-to-End Pipeline Orchestrator
==================================================

Connects all 8 agents across Person 1 and Person 2 into a single
deterministic pipeline:

    Complaint Text
        │
        ├─→ Person 1: CivicFlowAgent (Understanding → Classification)
        │       Output: domain_code, category_code, confidence
        │
        ├─→ Bridge: normalize_domain_name(domain_code)
        │
        ├─→ Person 2: RoutingAgent
        │       Output: department_code, category_code
        │
        ├─→ Person 2: PriorityAgent (DIPEX scoring)
        │       Output: priority, priority_score, sla_target_hours
        │
        ├─→ Person 2: SLAMonitoringAgent
        │       Output: sla_status, breach_probability
        │
        ├─→ Person 2: EscalationAgent
        │       Output: escalation_tier, action_required
        │
        └─→ Unified PipelineOutput + Learning Journal entry

Each stage is isolated via try/except so a failure in one agent does not
crash the entire pipeline. Partial results are always returned.
"""

import sys
import os
import traceback
from datetime import datetime, timezone
from typing import Optional

# Ensure project root is on path for cross-package imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pipeline.contracts import (
    PipelineInput,
    PipelineOutput,
    ClassificationResult,
    RoutingResult as PipelineRoutingResult,
    PriorityResult as PipelinePriorityResult,
    SLAResult,
    EscalationResult as PipelineEscalationResult,
)


class CivicFlowPipeline:
    """
    Unified orchestrator that chains Person 1 and Person 2 agents.

    Usage:
        pipeline = CivicFlowPipeline()
        result = pipeline.run("The streetlights on MG Road have been off for 3 days")
    """

    def __init__(self, learning_journal=None):
        """
        Initialize the pipeline.

        Args:
            learning_journal: Optional LearningJournal instance for feedback recording.
        """
        self._p1_agent = None
        self._learning_journal = learning_journal
        self._init_person1()

    def _init_person1(self):
        """Lazily initialize Person 1 agents."""
        try:
            from ai.agents.civicflow_agent import CivicFlowAgent
            self._p1_agent = CivicFlowAgent()
        except Exception as e:
            # Person 1 may not be importable in all environments
            self._p1_init_error = str(e)

    def run(self, input_data: PipelineInput) -> PipelineOutput:
        """
        Execute the full 8-agent pipeline.

        Args:
            input_data: PipelineInput with complaint text and optional metadata.

        Returns:
            PipelineOutput with results from all stages.
        """
        output = PipelineOutput(complaint_id=input_data.complaint_id)
        all_warnings = []

        # ──────────────────────────────────────────────────────────
        # STAGE 1: Person 1 — AI/NLP Classification
        # ──────────────────────────────────────────────────────────
        try:
            output = self._stage_classification(input_data, output)
            output.agent_chain.append("CivicFlowAgent")
            output.agent_chain.append("UnderstandingAgent")
            output.agent_chain.append("ClassificationAgent")
            output.agent_chain.append("UncertaintyGate")
        except Exception as e:
            output.errors.append(f"STAGE_1_CLASSIFICATION: {str(e)}")
            output.pipeline_status = "PARTIAL"

        # ──────────────────────────────────────────────────────────
        # STAGE 2: Bridge — Domain Normalization
        # ──────────────────────────────────────────────────────────
        try:
            output = self._stage_bridge(output)
        except Exception as e:
            output.errors.append(f"STAGE_2_BRIDGE: {str(e)}")

        # ──────────────────────────────────────────────────────────
        # STAGE 3: Person 2 — Routing Agent
        # ──────────────────────────────────────────────────────────
        try:
            output = self._stage_routing(input_data, output)
            output.agent_chain.append("RoutingAgent")
        except Exception as e:
            output.errors.append(f"STAGE_3_ROUTING: {str(e)}")
            output.pipeline_status = "PARTIAL"

        # ──────────────────────────────────────────────────────────
        # STAGE 4: Person 2 — Priority Agent (DIPEX)
        # ──────────────────────────────────────────────────────────
        try:
            output = self._stage_priority(input_data, output)
            output.agent_chain.append("PriorityAgent")
        except Exception as e:
            output.errors.append(f"STAGE_4_PRIORITY: {str(e)}")
            output.pipeline_status = "PARTIAL"

        # ──────────────────────────────────────────────────────────
        # STAGE 5: Person 2 — SLA Monitoring Agent
        # ──────────────────────────────────────────────────────────
        try:
            output = self._stage_sla(input_data, output)
            output.agent_chain.append("SLAMonitoringAgent")
        except Exception as e:
            output.errors.append(f"STAGE_5_SLA: {str(e)}")
            output.pipeline_status = "PARTIAL"

        # ──────────────────────────────────────────────────────────
        # STAGE 6: Person 2 — Escalation Agent
        # ──────────────────────────────────────────────────────────
        try:
            output = self._stage_escalation(input_data, output)
            output.agent_chain.append("EscalationAgent")
        except Exception as e:
            output.errors.append(f"STAGE_6_ESCALATION: {str(e)}")
            output.pipeline_status = "PARTIAL"

        # ──────────────────────────────────────────────────────────
        # STAGE 7: Learning Journal — Record outcome
        # ──────────────────────────────────────────────────────────
        if self._learning_journal:
            try:
                self._learning_journal.record_outcome(
                    complaint_text=input_data.complaint_text,
                    complaint_id=input_data.complaint_id,
                    predicted_domain=output.classification.normalized_domain or output.classification.domain_code,
                    predicted_category=output.classification.category_code,
                    confidence=output.classification.confidence,
                    routing_department=output.routing.department_code,
                    priority=output.priority.priority,
                    pipeline_status=output.pipeline_status,
                )
            except Exception:
                pass  # Learning journal failures must never crash the pipeline

        if not output.errors:
            output.pipeline_status = "SUCCESS"

        return output

    # ──────────────────────────────────────────────────────────────
    # Stage Implementations
    # ──────────────────────────────────────────────────────────────

    def _stage_classification(self, inp: PipelineInput, out: PipelineOutput) -> PipelineOutput:
        """Stage 1: Person 1 AI/NLP classification."""
        if self._p1_agent is None:
            raise RuntimeError(f"Person 1 agent not initialized: {getattr(self, '_p1_init_error', 'unknown')}")

        result = self._p1_agent.process_complaint(inp.complaint_text, debug=inp.debug)

        out.classification = ClassificationResult(
            status=result.get("status", "UNKNOWN"),
            domain_code=result.get("domain_code"),
            domain_name=result.get("domain_name"),
            category_code=result.get("category_code"),
            category_name=result.get("category_name"),
            department_code=result.get("department_code"),
            mapping_type=result.get("mapping_type"),
            confidence=result.get("confidence", 0.0),
            evidence=result.get("evidence", []),
            explanation=result.get("explanation"),
        )

        if inp.debug and "trace" in result:
            out.debug_trace = {"person1_trace": result["trace"]}

        return out

    def _stage_bridge(self, out: PipelineOutput) -> PipelineOutput:
        """Stage 2: Normalize Person 1 domain to Person 2 canonical name."""
        from civicflow_ai_person2_config import normalize_domain_name

        raw_domain = out.classification.domain_code
        if raw_domain:
            normalized = normalize_domain_name(raw_domain)
            out.classification.normalized_domain = normalized
            if normalized != raw_domain:
                out.warnings.append(
                    f"Domain aliased: '{raw_domain}' → '{normalized}'"
                )
        return out

    def _stage_routing(self, inp: PipelineInput, out: PipelineOutput) -> PipelineOutput:
        """Stage 3: Person 2 Routing Agent."""
        from civicflow_ai_person2_agents import route_complaint, RoutingRequest

        domain = out.classification.normalized_domain or out.classification.domain_code
        subcategory = out.classification.category_code

        request = RoutingRequest(
            complaint_id=inp.complaint_id,
            complaint_text=inp.complaint_text,
            existing_domain=domain,
            existing_subcategory=subcategory,
        )
        result = route_complaint(request)

        out.routing = PipelineRoutingResult(
            status=result.status,
            domain=result.domain,
            subcategory=result.subcategory,
            category_code=result.category_code,
            department_code=result.department_code,
            confidence=result.confidence,
            reason=result.reason,
            category_mapping_available=result.category_mapping_available,
            warnings=result.warnings,
        )
        out.warnings.extend(result.warnings)
        return out

    def _stage_priority(self, inp: PipelineInput, out: PipelineOutput) -> PipelineOutput:
        """Stage 4: Person 2 Priority Agent (DIPEX scoring)."""
        from civicflow_ai_person2_agents import evaluate_priority, PriorityRequest

        request = PriorityRequest(
            complaint_id=inp.complaint_id,
            complaint_text=inp.complaint_text,
            domain=out.routing.domain,
            subcategory=out.routing.subcategory,
            category_code=out.routing.category_code,
            location_context=inp.location_context,
            affected_count=inp.affected_count,
            duration_days=inp.duration_days,
            recurrence_count=inp.recurrence_count,
        )
        result = evaluate_priority(request)

        out.priority = PipelinePriorityResult(
            status=result.status,
            priority=result.priority,
            priority_score=result.priority_score,
            sla_target_hours=result.sla_target_hours,
            factors=result.factors.model_dump() if result.factors else None,
            override_applied=result.override_applied,
            override_reason=result.override_reason,
            confidence=result.confidence,
            explanation=result.explanation,
            risk_factors=result.risk_factors,
            warnings=result.warnings,
        )
        out.warnings.extend(result.warnings)
        return out

    def _stage_sla(self, inp: PipelineInput, out: PipelineOutput) -> PipelineOutput:
        """Stage 5: Person 2 SLA Monitoring Agent."""
        from civicflow_ai_person2_agents import evaluate_sla_status, SLAMonitorRequest

        now = datetime.now(timezone.utc)
        request = SLAMonitorRequest(
            ticket_id=inp.complaint_id or "PIPELINE-EVAL",
            priority=out.priority.priority or "MEDIUM",
            category_code=out.routing.category_code,
            subcategory=out.routing.subcategory,
            status="ASSIGNED",
            created_at=now,
            current_time=now,
            sla_target_hours=out.priority.sla_target_hours,
        )
        result = evaluate_sla_status(request)

        out.sla = SLAResult(
            status=result.status,
            sla_status=result.sla_status,
            sla_tier=result.sla_tier,
            elapsed_hours=result.elapsed_hours,
            remaining_hours=result.remaining_hours,
            sla_target_hours=result.sla_target_hours,
            sla_ratio=result.sla_ratio,
            is_breached=result.is_breached,
            breach_probability=result.breach_probability,
            explanation=result.explanation,
            warnings=result.warnings,
        )
        out.warnings.extend(result.warnings)
        return out

    def _stage_escalation(self, inp: PipelineInput, out: PipelineOutput) -> PipelineOutput:
        """Stage 6: Person 2 Escalation Agent."""
        from civicflow_ai_person2_agents import evaluate_escalation, EscalationRequest

        request = EscalationRequest(
            ticket_id=inp.complaint_id or "PIPELINE-EVAL",
            complaint_text=inp.complaint_text,
            department_code=out.routing.department_code or "SAN",
            priority=out.priority.priority or "MEDIUM",
            sla_status=out.sla.sla_status or "WITHIN_SLA",
            sla_ratio=out.sla.sla_ratio,
            sla_tier=out.sla.sla_tier,
            is_breached=out.sla.is_breached,
            breach_probability=out.sla.breach_probability,
            affected_count=inp.affected_count,
        )
        result = evaluate_escalation(request)

        out.escalation = PipelineEscalationResult(
            status=result.status,
            escalation_tier=result.escalation_tier,
            action_required=result.action_required,
            target_role=result.target_role,
            target_contact=result.target_contact,
            recommended_officer_id=result.recommended_officer_id,
            recommended_officer_name=result.recommended_officer_name,
            penalty_score=result.penalty_score,
            notification_draft=result.notification_draft,
            explanation=result.explanation,
            warnings=result.warnings,
        )
        out.warnings.extend(result.warnings)
        return out
