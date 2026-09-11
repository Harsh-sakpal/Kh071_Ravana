"""
CivicFlow AI — Self-Repairing Test Harness
=============================================

Self-healing infrastructure that:
  - Auto-detects taxonomy drift between Person 1 and Person 2
  - Validates all alias mappings are bidirectionally consistent
  - Flags unmapped domains and auto-generates fix suggestions
  - Validates Person 2 agent chain structural contracts
  - Retry decorator for flaky integration points
"""

import sys
import os
import time
import functools
from typing import Callable, Any

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2_ROOT = os.path.join(PROJECT_ROOT, "civicflow-ai-person2")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, P2_ROOT)


# ──────────────────────────────────────────────────────────────────────────
# Retry Decorator (Self-Repair Utility)
# ──────────────────────────────────────────────────────────────────────────

def retry_on_failure(max_retries: int = 3, backoff_base: float = 0.1):
    """
    Decorator that retries a test function on failure with exponential backoff.
    Useful for integration tests that may have transient failures.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait = backoff_base * (2 ** attempt)
                        time.sleep(wait)
            raise last_exception
        return wrapper
    return decorator


# ──────────────────────────────────────────────────────────────────────────
# Taxonomy Drift Detection
# ──────────────────────────────────────────────────────────────────────────

class TestTaxonomyDrift:
    """
    Auto-detects when Person 1 and Person 2 taxonomies diverge.
    If Person 1 adds a new domain, these tests will catch the gap.
    """

    def _get_p1_domains(self):
        """Get Person 1's declared fixed domains."""
        from ai.taxonomy.taxonomy_manager import TaxonomyManager
        return set(TaxonomyManager.FIXED_DOMAINS)

    def _get_p2_domains(self):
        """Get Person 2's canonical taxonomy domains."""
        from app.config.taxonomy import TAXONOMY
        return set(TAXONOMY.keys())

    def _get_alias_map(self):
        from app.config.taxonomy import DOMAIN_ALIASES
        return DOMAIN_ALIASES

    def test_all_p1_domains_are_mapped(self):
        """Every Person 1 domain must either exist in P2 or have an alias."""
        p1_domains = self._get_p1_domains()
        p2_domains = self._get_p2_domains()
        aliases = self._get_alias_map()

        unmapped = []
        for p1_domain in p1_domains:
            if p1_domain not in p2_domains and p1_domain not in aliases:
                unmapped.append(p1_domain)

        if unmapped:
            # Self-repair: generate fix suggestion
            suggestion = self._generate_alias_suggestions(unmapped, p2_domains)
            pytest.fail(
                f"TAXONOMY DRIFT DETECTED: {len(unmapped)} Person 1 domain(s) have no "
                f"Person 2 mapping or alias.\n"
                f"Unmapped domains: {unmapped}\n"
                f"Suggested fixes:\n{suggestion}"
            )

    def test_no_orphaned_aliases(self):
        """Every alias must point to a valid P2 domain."""
        p2_domains = self._get_p2_domains()
        aliases = self._get_alias_map()

        orphaned = []
        for alias, target in aliases.items():
            if target not in p2_domains:
                orphaned.append((alias, target))

        if orphaned:
            pytest.fail(
                f"ORPHANED ALIASES: {len(orphaned)} alias(es) point to non-existent "
                f"Person 2 domains:\n{orphaned}\n"
                f"Fix: Update DOMAIN_ALIASES in taxonomy.py"
            )

    def test_alias_coverage_complete(self):
        """
        Every P1 domain that doesn't exist verbatim in P2 must have an alias.
        This catches 'silent' domain name changes.
        """
        p1_domains = self._get_p1_domains()
        p2_domains = self._get_p2_domains()
        aliases = self._get_alias_map()

        missing_aliases = []
        for p1_domain in p1_domains:
            if p1_domain not in p2_domains:
                if p1_domain not in aliases:
                    missing_aliases.append(p1_domain)

        assert len(missing_aliases) == 0, (
            f"Missing alias entries for P1 domains not in P2: {missing_aliases}"
        )

    def test_reverse_alias_consistency(self):
        """No two aliases should map to the same P2 domain."""
        aliases = self._get_alias_map()
        seen_targets = {}
        duplicates = []
        for alias, target in aliases.items():
            if target in seen_targets:
                duplicates.append((alias, seen_targets[target], target))
            seen_targets[target] = alias

        assert len(duplicates) == 0, (
            f"Multiple aliases map to the same P2 domain: {duplicates}"
        )

    def _generate_alias_suggestions(self, unmapped: list, p2_domains: set) -> str:
        """Auto-generate fix suggestions based on string similarity."""
        suggestions = []
        for domain in unmapped:
            # Simple substring matching for suggestions
            candidates = []
            for p2 in p2_domains:
                # Check if P1 domain name is a substring/prefix of P2 domain
                if domain in p2 or p2.startswith(domain.split("_")[0]):
                    candidates.append(p2)

            if candidates:
                suggestions.append(
                    f"  '{domain}' → possibly '{candidates[0]}'\n"
                    f"    Add to DOMAIN_ALIASES: \"{domain}\": \"{candidates[0]}\","
                )
            else:
                suggestions.append(
                    f"  '{domain}' → NO MATCH FOUND (new P2 domain may be needed)"
                )
        return "\n".join(suggestions)


# ──────────────────────────────────────────────────────────────────────────
# DB Mapping Self-Repair
# ──────────────────────────────────────────────────────────────────────────

class TestDBMappingSelfRepair:
    """Validates DB mapping consistency and flags gaps."""

    def test_all_p2_domains_have_department(self):
        from app.config.taxonomy import TAXONOMY
        from app.config.db_mapping import department_for_domain

        unmapped = []
        for domain in TAXONOMY.keys():
            if department_for_domain(domain) is None:
                unmapped.append(domain)

        assert len(unmapped) == 0, (
            f"Domains without department mapping: {unmapped}\n"
            f"Fix: Add entries to DOMAIN_TO_DEPARTMENT in db_mapping.py"
        )

    def test_all_aliases_have_department(self):
        from app.config.taxonomy import DOMAIN_ALIASES
        from app.config.db_mapping import department_for_domain

        unmapped = []
        for alias in DOMAIN_ALIASES.keys():
            if department_for_domain(alias) is None:
                unmapped.append(alias)

        assert len(unmapped) == 0, (
            f"Aliases without department mapping: {unmapped}\n"
            f"Fix: Add alias entries to DOMAIN_TO_DEPARTMENT in db_mapping.py"
        )

    def test_department_codes_valid(self):
        from app.config.db_mapping import DOMAIN_TO_DEPARTMENT

        valid_depts = {"SAN", "RDW", "ELE", "WTR", "PRK"}
        invalid = {
            domain: dept
            for domain, dept in DOMAIN_TO_DEPARTMENT.items()
            if dept not in valid_depts
        }
        assert len(invalid) == 0, (
            f"Invalid department codes: {invalid}\n"
            f"Valid codes: {valid_depts}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Agent Contract Self-Repair
# ──────────────────────────────────────────────────────────────────────────

class TestAgentContractIntegrity:
    """
    Validates that all Person 2 agents are importable, callable,
    and produce structurally valid outputs.
    """

    @retry_on_failure(max_retries=2)
    def test_routing_agent_importable_and_callable(self):
        from app.agents.routing_agent import route_complaint
        from app.models.routing import RoutingRequest

        req = RoutingRequest(complaint_text="Test complaint for import validation")
        result = route_complaint(req)
        assert hasattr(result, "status")
        assert hasattr(result, "domain")
        assert hasattr(result, "department_code")

    @retry_on_failure(max_retries=2)
    def test_priority_agent_importable_and_callable(self):
        from app.agents.priority_agent import evaluate_priority
        from app.models.priority import PriorityRequest

        req = PriorityRequest(complaint_text="Test complaint for priority validation")
        result = evaluate_priority(req)
        assert hasattr(result, "priority")
        assert hasattr(result, "priority_score")
        assert hasattr(result, "factors")

    @retry_on_failure(max_retries=2)
    def test_sla_agent_importable_and_callable(self):
        from app.agents.sla_monitoring_agent import evaluate_sla_status
        from app.models.sla_monitoring import SLAMonitorRequest
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        req = SLAMonitorRequest(
            ticket_id="SELF-REPAIR-SLA",
            created_at=now,
            current_time=now,
        )
        result = evaluate_sla_status(req)
        assert hasattr(result, "sla_status")
        assert hasattr(result, "breach_probability")

    @retry_on_failure(max_retries=2)
    def test_escalation_agent_importable_and_callable(self):
        from app.agents.escalation_agent import evaluate_escalation
        from app.models.escalation import EscalationRequest

        req = EscalationRequest(
            ticket_id="SELF-REPAIR-ESC",
            sla_ratio=0.5,
            breach_probability=0.3,
        )
        result = evaluate_escalation(req)
        assert hasattr(result, "escalation_tier")
        assert hasattr(result, "action_required")

    def test_agent_output_chain_compatibility(self):
        """Verify that Routing output fields feed into Priority input fields."""
        from app.models.routing import RoutingResult
        from app.models.priority import PriorityRequest

        routing_fields = set(RoutingResult.model_fields.keys())
        priority_inputs = {"domain", "subcategory", "category_code"}

        missing = priority_inputs - routing_fields
        assert len(missing) == 0, (
            f"Priority Agent expects fields from Routing that don't exist: {missing}\n"
            f"Fix: Add these fields to RoutingResult in routing.py"
        )
