"""
Agentic Classification Agent for CivicFlow AI (Phase 4).

Features:
- Conservative Agent Safety Rule: Always accepts high-confidence TF-IDF predictions to eliminate regressions.
- Dedicated Uncertainty & Adversarial Out-Of-Scope Gate pre-filter.
- Enhanced trace metadata capturing tool scores, confidence, margins, thresholds, decisions, and reasons.
"""

from typing import Dict, Any, List, Optional
from ai.schemas.dataclasses import (
    ClassificationOutput,
    SemanticClassification,
    AdministrativeMapping,
    ClassificationStatus,
    MappingType,
    ClassificationMethod,
    UnderstandingOutput,
)
from ai.taxonomy.taxonomy_manager import TaxonomyManager
from ai.agents.uncertainty_gate import UncertaintyGate
from ai.tools.tfidf_tool import TfidfClassificationTool
from ai.tools.embedding_tool import EmbeddingSimilarityTool
from ai.tools.llm_tool import LlmFallbackTool


class ClassificationAgent:
    """
    Agentic Classification Agent managing confidence-based tool escalation with regression protection.
    """

    # Phase 4 Tuned Default Thresholds (validated on validation.csv)
    DEFAULT_TFIDF_THRESHOLD = 0.80
    DEFAULT_TFIDF_MARGIN = 0.15
    DEFAULT_EMBEDDING_THRESHOLD = 0.70

    def __init__(
        self,
        taxonomy_manager: Optional[TaxonomyManager] = None,
        tfidf_threshold: float = DEFAULT_TFIDF_THRESHOLD,
        tfidf_margin: float = DEFAULT_TFIDF_MARGIN,
        embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
    ):
        self.taxonomy_manager = taxonomy_manager or TaxonomyManager()
        self.tfidf_threshold = tfidf_threshold
        self.tfidf_margin = tfidf_margin
        self.embedding_threshold = embedding_threshold

        self.uncertainty_gate = UncertaintyGate()
        self.tfidf_tool = TfidfClassificationTool(taxonomy_manager=self.taxonomy_manager)
        
        try:
            self.embedding_tool = EmbeddingSimilarityTool(taxonomy_manager=self.taxonomy_manager)
        except Exception:
            self.embedding_tool = None

        self.llm_tool = LlmFallbackTool(taxonomy_manager=self.taxonomy_manager)

    def classify(self, understanding: UnderstandingOutput, debug: bool = True) -> ClassificationOutput:
        """
        Main entrypoint: Accepts structured UnderstandingOutput and returns ClassificationOutput
        with enhanced explainable execution trace.
        """
        trace = []
        text = understanding.original_text or ""
        norm_text = understanding.normalized_text or ""

        # 1. Pre-classification Uncertainty & Out-Of-Scope Gate
        if self.uncertainty_gate:
            gate_status, gate_reason = self.uncertainty_gate.evaluate(text, norm_text)
            if gate_status:
                trace.append({
                    "tool": "UNCERTAINTY_GATE",
                    "confidence": 0.95,
                    "threshold": 0.90,
                    "decision": "ACCEPT",
                    "reason": gate_reason,
                    "final_selection": None
                })
                out = ClassificationOutput(
                    status=gate_status,
                    semantic_classification=SemanticClassification(),
                    administrative_mapping=AdministrativeMapping(type=MappingType.UNMAPPED.value, department_code=None),
                    overall_confidence=0.95,
                    method=ClassificationMethod.RULE_ENGINE.value,
                    evidence=[gate_reason],
                )
                if debug:
                    out.secondary_signals = [{"trace": trace}]
                return out

        # 2. STEP 1: Execute TF-IDF Tool
        tfidf_res = self.tfidf_tool.classify(norm_text)
        tfidf_conf = tfidf_res.get("overall_confidence", 0.0)
        tfidf_preds = tfidf_res.get("top_predictions", [])

        tfidf_margin = 0.0
        if len(tfidf_preds) > 1:
            tfidf_margin = tfidf_preds[0]["score"] - tfidf_preds[1]["score"]

        # AGENT SAFETY RULE: High Confidence & Margin -> ACCEPT TF-IDF (unless semantic contradiction detected)
        tfidf_cat = tfidf_res.get("category", {}).get("code")
        has_tfidf_contradiction = self._has_semantic_contradiction(text, tfidf_cat)

        if tfidf_conf >= self.tfidf_threshold and tfidf_margin >= self.tfidf_margin and not has_tfidf_contradiction:
            trace.append({
                "tool": "TFIDF",
                "raw_score": tfidf_preds[0]["score"] if tfidf_preds else 0.0,
                "confidence": tfidf_conf,
                "margin": round(tfidf_margin, 4),
                "threshold": self.tfidf_threshold,
                "decision": "ACCEPT",
                "reason": "Agent Safety Rule: High TF-IDF confidence and strong score margin",
                "final_selection": tfidf_res["category"]["code"]
            })
            out = self._build_classification_output(tfidf_res, method="TFIDF", evidence=["TF-IDF early accept (Safety Rule)"])
            if debug:
                out.secondary_signals = [{"trace": trace}]
            return out

        trace.append({
            "tool": "TFIDF",
            "raw_score": tfidf_preds[0]["score"] if tfidf_preds else 0.0,
            "confidence": tfidf_conf,
            "margin": round(tfidf_margin, 4),
            "threshold": self.tfidf_threshold,
            "decision": "ESCALATE",
            "reason": "Semantic contradiction detected" if has_tfidf_contradiction else f"Confidence ({tfidf_conf:.2f}) < {self.tfidf_threshold} or margin ({tfidf_margin:.2f}) < {self.tfidf_margin}",
            "final_selection": None
        })

        # 3. STEP 2: Escalate to Embedding Tool (if available)
        embedding_res = None
        if self.embedding_tool and self.embedding_tool.is_ready:
            try:
                embedding_res = self.embedding_tool.classify(norm_text)
                emb_conf = embedding_res.get("overall_confidence", 0.0)
                raw_sim = embedding_res.get("raw_similarity", 0.0)
                emb_cat = embedding_res.get("category", {}).get("code")
                has_emb_contradiction = self._has_semantic_contradiction(text, emb_cat)

                if emb_conf >= self.embedding_threshold and not has_emb_contradiction:
                    trace.append({
                        "tool": "EMBEDDING",
                        "raw_score": raw_sim,
                        "confidence": emb_conf,
                        "margin": embedding_res.get("margin", 0.0),
                        "threshold": self.embedding_threshold,
                        "decision": "ACCEPT",
                        "reason": "High vector embedding similarity confidence",
                        "final_selection": embedding_res["category"]["code"]
                    })
                    out = self._build_classification_output(embedding_res, method="EMBEDDING", evidence=["Vector embedding similarity match"])
                    if debug:
                        out.secondary_signals = [{"trace": trace}]
                    return out

                trace.append({
                    "tool": "EMBEDDING",
                    "raw_score": raw_sim,
                    "confidence": emb_conf,
                    "margin": embedding_res.get("margin", 0.0),
                    "threshold": self.embedding_threshold,
                    "decision": "ESCALATE",
                    "reason": "Semantic contradiction detected" if has_emb_contradiction else f"Embedding confidence ({emb_conf:.2f}) < {self.embedding_threshold}",
                    "final_selection": None
                })
            except Exception as e:
                trace.append({"tool": "EMBEDDING", "decision": "ERROR", "error": str(e)})

        # 4. STEP 3: Escalate to Gemini Semantic Resolver & Taxonomy Fit Check
        try:
            from ai.tools.gemini_semantic_resolver import GeminiSemanticResolver
            from ai.tools.taxonomy_fit_checker import TaxonomyFitChecker

            semantic_resolver = GeminiSemanticResolver(taxonomy_manager=self.taxonomy_manager)
            fit_checker = TaxonomyFitChecker(taxonomy_manager=self.taxonomy_manager)

            resolved_intent = semantic_resolver.resolve_intent(text)
            if resolved_intent:
                fit_result = fit_checker.check_fit(resolved_intent, original_text=text)
                llm_res = {
                    "status": fit_result["status"],
                    "domain": {"code": fit_result.get("domain_code"), "confidence": fit_result.get("confidence", 0.90)},
                    "category": {"code": fit_result.get("category_code"), "confidence": fit_result.get("confidence", 0.90)},
                    "administrative_mapping": fit_result.get("administrative_mapping", {"type": MappingType.UNMAPPED.value, "department_code": None}),
                    "overall_confidence": fit_result.get("confidence", 0.90),
                    "reasoning": fit_result.get("explanation", "Gemini semantic resolver and taxonomy fit check."),
                    "evidence": fit_result.get("evidence", [])
                }
                llm_conf = llm_res.get("overall_confidence", 0.90)
                llm_cat = fit_result.get("category_code")
                trace.append({
                    "tool": "GEMINI_SEMANTIC_RESOLVER",
                    "raw_score": llm_conf,
                    "confidence": llm_conf,
                    "margin": 0.0,
                    "threshold": 0.50,
                    "decision": "ACCEPT",
                    "reason": llm_res.get("reasoning"),
                    "final_selection": llm_cat
                })
            else:
                # Fallback to existing LLM tool
                llm_res = self.llm_tool.classify(
                    text=norm_text,
                    understanding=understanding.to_dict(),
                    tfidf_candidates=tfidf_preds,
                    embedding_candidates=embedding_res.get("top_predictions") if embedding_res else []
                )
                llm_conf = llm_res.get("overall_confidence", 0.0)
                llm_cat = llm_res.get("category", {}).get("code")
                trace.append({
                    "tool": "LLM_FALLBACK",
                    "raw_score": llm_conf,
                    "confidence": llm_conf,
                    "margin": 0.0,
                    "threshold": 0.50,
                    "decision": "ACCEPT",
                    "reason": llm_res.get("reasoning", "LLM fallback taxonomy-constrained resolution"),
                    "final_selection": llm_cat
                })
        except Exception as e:
            # Task F: Failure Recovery
            trace.append({"tool": "GEMINI_SEMANTIC_RESOLVER", "decision": "ERROR", "error": str(e)})
            llm_res = {
                "status": ClassificationStatus.AMBIGUOUS.value,
                "overall_confidence": 0.0,
                "reasoning": "LLM fallback failed due to internal error."
            }

        out = self._build_classification_output(llm_res, method="LLM_FALLBACK", evidence=llm_res.get("evidence") or [llm_res.get("reasoning", "LLM fallback resolution")])
        if debug:
            out.secondary_signals = [{"trace": trace}]
        return out

    def _build_classification_output(self, tool_result: Dict[str, Any], method: str, evidence: List[str]) -> ClassificationOutput:
        """Helper building standardized ClassificationOutput dataclass from tool result."""
        status = tool_result.get("status", ClassificationStatus.CLASSIFIED.value)
        dom_data = tool_result.get("domain", {})
        cat_data = tool_result.get("category", {})
        dept_map_data = tool_result.get("administrative_mapping", {})

        dom_code = dom_data.get("code") if isinstance(dom_data, dict) else dom_data
        cat_code = cat_data.get("code") if isinstance(cat_data, dict) else cat_data
        cat_name = cat_data.get("name") if isinstance(cat_data, dict) else None

        dom_info = self.taxonomy_manager.get_domain(dom_code) if dom_code else None
        dom_name = dom_info.get("domain_name") if dom_info else None

        if not cat_name and cat_code:
            cat_info = self.taxonomy_manager.get_category(cat_code)
            if cat_info:
                cat_name = cat_info.get("category_name")

        sem_class = SemanticClassification(
            domain_code=dom_code,
            domain_name=dom_name,
            category_code=cat_code,
            category_name=cat_name,
        )

        admin_map = AdministrativeMapping(
            type=dept_map_data.get("type", MappingType.DIRECT.value),
            department_code=dept_map_data.get("department_code"),
            notes=dept_map_data.get("notes"),
        )

        return ClassificationOutput(
            status=status,
            semantic_classification=sem_class,
            administrative_mapping=admin_map,
            overall_confidence=tool_result.get("overall_confidence", 0.0),
            method=method,
            evidence=evidence,
            explanation=tool_result.get("reasoning", "Classified based on taxonomy matching"),
        )

    def _has_semantic_contradiction(self, original_text: str, predicted_category: Optional[str]) -> bool:
        """Detects if raw classifier predicted category contradicts substance/entity in text."""
        if not original_text or not predicted_category:
            return False

        text_lower = original_text.lower()
        if predicted_category.startswith("WTR-") or predicted_category == "WTR-LEAK":
            non_water_indicators = [
                "oil pipe", "oil pipeline", "oil transport", "oil leak", "oil leakage",
                "gas pipe", "gas pipeline", "gas leak", "chemical pipe", "chemical pipeline",
                "tel nighayla", "तेल", "ऑईल", "गॅस", "तेलाच्या", "केमिकल", "विद्युत", "विदयुत",
                "telachya", "telatun", "telacha", "vidyut", "current"
            ]
            if any(kw in text_lower for kw in non_water_indicators):
                return True
            import re
            if re.search(r"\b(oil|oils|gas|gases|petroleum|chemical|chemicals|diesel|fuel|kerosene|electricity|electrical|electric|current|vidyut)\b", text_lower):
                return True

        return False
