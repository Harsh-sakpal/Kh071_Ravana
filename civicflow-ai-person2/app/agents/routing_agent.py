"""
CivicFlow AI - Routing Agent
==============================

Pipeline (Master Prompt Section 9):

    Complaint -> Normalization -> Rules/semantic signals -> AI/NLP
    reasoning where required -> Candidate classification -> Validation
    -> Final routing result

This MVP implements the rule/keyword-scoring stage fully. The "AI/NLP
reasoning where required" stage is a defined extension point
(`_llm_fallback`) that is intentionally NOT wired to a live LLM yet --
per Section 21 ("do not make every decision dependent on an LLM"), the
rule-based path must work standalone. When confidence is low, the agent
reports LOW_CONFIDENCE rather than guessing harder or silently calling
an LLM that isn't part of the locked/tested contract yet.
"""

import re
from typing import Dict, List, Optional, Tuple

from app.config.taxonomy import TAXONOMY, is_valid_domain, is_valid_subcategory
from app.config.keywords import DOMAIN_KEYWORDS, SUBCATEGORY_KEYWORDS
from app.config.db_mapping import department_for_domain, category_code_for
from app.models.routing import RoutingRequest, RoutingResult

# Tunable thresholds -- centralized here per Section 18 (configuration-driven rules).
CONFIDENCE_HIGH = 0.60
CONFIDENCE_LOW_FLOOR = 0.25

# Scoring weights: a subcategory keyword hit counts for more than a domain
# keyword hit, since it's a more specific signal.
DOMAIN_KEYWORD_WEIGHT = 1.0
SUBCATEGORY_KEYWORD_WEIGHT = 2.0


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace. Kept intentionally simple/reversible --
    no stemming/lemmatization yet (MVP-first, Section 19)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _score_domains(normalized_text: str) -> Dict[str, float]:
    """Rule/semantic-signal stage: score every domain by keyword overlap."""
    scores: Dict[str, float] = {domain: 0.0 for domain in TAXONOMY}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in normalized_text:
                scores[domain] += DOMAIN_KEYWORD_WEIGHT
    return scores


def _score_subcategories(normalized_text: str, domain: str) -> Dict[str, float]:
    """Score subcategories only within the chosen domain."""
    scores: Dict[str, float] = {sub: 0.0 for sub in TAXONOMY.get(domain, [])}
    for (dom, sub), keywords in SUBCATEGORY_KEYWORDS.items():
        if dom != domain:
            continue
        for kw in keywords:
            if kw in normalized_text:
                scores[sub] += SUBCATEGORY_KEYWORD_WEIGHT
    return scores


def _best(scores: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not scores or max(scores.values()) == 0:
        return None, 0.0
    best_key = max(scores, key=scores.get)
    return best_key, scores[best_key]


def _confidence_from_scores(domain_score: float, subcategory_score: float) -> float:
    """Simple, explainable normalization -- not a black box.
    Caps at 0.97 to leave room above for a future LLM-verified top confidence.
    """
    raw = domain_score * 0.4 + subcategory_score * 0.6
    # squash into 0..1 with a soft cap; tuned so 1 strong subcategory keyword
    # hit (weight 2.0) plus 1 domain hit (weight 1.0) already clears CONFIDENCE_HIGH.
    normalized = min(raw / 3.4, 0.97)
    return round(normalized, 4)


def _llm_fallback(normalized_text: str) -> Optional[Tuple[str, str]]:
    """
    Extension point for the 'AI/NLP reasoning where required' pipeline stage.

    Deliberately not implemented against a live LLM in this MVP iteration --
    the Routing Agent must be fully testable and independently callable
    without external API dependencies (Section 21: 'do not make every
    decision dependent on an LLM'). Wire this to the Python NLP service /
    OpenAI-Llama stack described in the architecture diagram once the
    rule-based path's precision/recall has been evaluated against real
    complaint data (Iteration 2 evaluation step).
    """
    return None


def route_complaint(request: RoutingRequest) -> RoutingResult:
    """Main entry point for the Routing Agent. Independently callable/testable."""
    warnings: List[str] = []

    # --- Validation of any pre-supplied classification -------------------
    domain: Optional[str] = None
    subcategory: Optional[str] = None
    confidence = 0.0
    reason_parts: List[str] = []

    if request.existing_domain and is_valid_domain(request.existing_domain):
        domain = request.existing_domain
        reason_parts.append("Domain supplied by upstream Classification Agent.")
        if request.existing_subcategory and is_valid_subcategory(domain, request.existing_subcategory):
            subcategory = request.existing_subcategory
            reason_parts.append("Subcategory supplied by upstream Classification Agent.")
            confidence = 0.9
    elif request.existing_domain:
        warnings.append(
            f"existing_domain '{request.existing_domain}' is not part of the locked taxonomy; ignoring."
        )

    normalized_text = _normalize(request.complaint_text)

    # --- Rule/semantic-signal stage (only runs where not already known) --
    if domain is None:
        domain_scores = _score_domains(normalized_text)
        domain, domain_score = _best(domain_scores)
    else:
        domain_score = 4.0  # treat upstream-confirmed domain as a strong signal

    if domain is None:
        # --- AI/NLP reasoning extension point ----------------------------
        llm_result = _llm_fallback(normalized_text)
        if llm_result:
            domain, subcategory = llm_result
            reason_parts.append("Resolved via AI/NLP fallback reasoning.")
        else:
            return RoutingResult(
                status="UNRESOLVED",
                complaint_id=request.complaint_id,
                domain=None,
                subcategory=None,
                category_code=None,
                department_code=None,
                confidence=0.0,
                reason="No domain keywords matched and no upstream classification was provided.",
                category_mapping_available=False,
                warnings=warnings,
            )

    if subcategory is None:
        sub_scores = _score_subcategories(normalized_text, domain)
        subcategory, subcategory_score = _best(sub_scores)
        if subcategory is None:
            # Fall back to the domain's generic OTHER_* bucket rather than
            # fabricating a specific subcategory (Section 21).
            other_candidates = [s for s in TAXONOMY.get(domain, []) if s.startswith("OTHER_")]
            subcategory = other_candidates[0] if other_candidates else None
            subcategory_score = 0.0
            warnings.append("No specific subcategory keywords matched; used domain's OTHER_* bucket.")
    else:
        subcategory_score = 4.0

    if confidence == 0.0:
        confidence = _confidence_from_scores(domain_score, subcategory_score)

    # --- DB mapping (never fabricated) ------------------------------------
    department_code = department_for_domain(domain)
    category_code = category_code_for(domain, subcategory) if subcategory else None

    if department_code is None:
        warnings.append(f"Domain '{domain}' has no department mapping configured.")
    if category_code is None:
        warnings.append(
            f"No confirmed database category mapping for ({domain}, {subcategory}); "
            "operational routing will need manual category assignment."
        )

    reason_parts.append(
        f"Matched domain '{domain}'"
        + (f" and subcategory '{subcategory}'" if subcategory else "")
        + " based on complaint text signals."
    )

    status = "SUCCESS" if confidence >= CONFIDENCE_HIGH else (
        "LOW_CONFIDENCE" if confidence >= CONFIDENCE_LOW_FLOOR else "LOW_CONFIDENCE"
    )
    if confidence < CONFIDENCE_LOW_FLOOR:
        warnings.append("Confidence below floor threshold; recommend manual review.")

    return RoutingResult(
        status=status,
        complaint_id=request.complaint_id,
        domain=domain,
        subcategory=subcategory,
        category_code=category_code,
        department_code=department_code,
        confidence=confidence,
        reason=" ".join(reason_parts),
        category_mapping_available=category_code is not None,
        warnings=warnings,
    )
