"""
Taxonomy-Constrained LLM Fallback Tool for CivicFlow AI.

Provides structured LLM classification when lexical (TF-IDF) and vector (Embedding) tools
express high uncertainty. Strictly constrained to domain and category codes defined in taxonomy.json.

Supports:
- Google Gemini API (primary, via GEMINI_API_KEY)
- Offline mock consensus (fallback when no API key is set)
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from ai.taxonomy.taxonomy_manager import TaxonomyManager


class LlmFallbackTool:
    """Taxonomy-constrained LLM Fallback classification tool."""

    PROMPT_TEMPLATE = """You are the Municipal Complaint Classification Assistant for CivicFlow AI.
Your task is to classify the citizen complaint into EXACTLY ONE valid domain and category from the provided taxonomy.

STRICT TAXONOMY RULES:
- You MUST ONLY choose domain_code from the allowed 15 AI domains:
{allowed_domains}

- You MUST ONLY choose category_code from the allowed category codes for that domain:
{allowed_categories}

- Allowed Status values: ["CLASSIFIED", "AMBIGUOUS", "UNKNOWN", "OUT_OF_SCOPE", "INSUFFICIENT_INFORMATION"]

CLASSIFICATION RULES:
- If the complaint is clearly about a municipal civic issue, set status to "CLASSIFIED" and pick the best matching domain and category.
- If the complaint is vague with multiple conflicting topics, set status to "AMBIGUOUS".
- If the complaint is completely unrelated to municipal civic issues (e.g., tech support, entertainment, academics, personal shopping), set status to "OUT_OF_SCOPE".
- If the complaint is too short or lacks actionable detail, set status to "INSUFFICIENT_INFORMATION".
- If you genuinely cannot determine the category, set status to "UNKNOWN".

COMPLAINT DETAILS:
Original Text: "{original_text}"
Normalized Text: "{normalized_text}"
Detected Language: {language}
Extracted Signals: {signals}

TOP TOOL CANDIDATES:
TF-IDF Candidates: {tfidf_candidates}
Embedding Candidates: {embedding_candidates}

OUTPUT FORMAT INSTRUCTIONS:
Return ONLY a valid JSON object with NO additional text, NO markdown, NO code fences:
{{
  "status": "CLASSIFIED",
  "domain_code": "<DOMAIN_CODE>",
  "category_code": "<CATEGORY_CODE>",
  "confidence": 0.85,
  "evidence": ["<keyword1>", "<keyword2>"]
}}

If status is not "CLASSIFIED", set domain_code and category_code to null.
"""

    def __init__(self, api_key: Optional[str] = None, taxonomy_manager: Optional[TaxonomyManager] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.tm = taxonomy_manager or TaxonomyManager()
        self.mock_mode = os.getenv("CIVICFLOW_MOCK_LLM", "false").lower() == "true" or self.api_key is None
        self._gemini_model = None

    def _get_gemini_model(self):
        """Lazily initialize the Gemini model to avoid import cost on every request."""
        if self._gemini_model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._gemini_model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_output_tokens": 512,
                    "response_mime_type": "application/json",
                },
            )
        return self._gemini_model

    def build_prompt(
        self,
        text: str,
        understanding: Optional[Dict[str, Any]] = None,
        tfidf_candidates: Optional[List[Dict[str, Any]]] = None,
        embedding_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Constructs taxonomy-constrained LLM prompt."""
        domains = self.tm.get_domains()
        categories = self.tm.get_categories()

        allowed_domains_str = json.dumps(domains, indent=2)
        allowed_cats_str = json.dumps(
            {code: {"name": info["category_name"], "domain": info.get("domain_code")} for code, info in categories.items()},
            indent=2,
        )

        norm_text = text
        lang = "EN"
        signals = {}
        if understanding:
            if hasattr(understanding, "normalized_text"):
                norm_text = understanding.normalized_text
                lang = understanding.language.code
                signals = understanding.signals
            elif isinstance(understanding, dict):
                norm_text = understanding.get("normalized_text", text)
                lang = understanding.get("language", {}).get("code", "EN")
                signals = understanding.get("signals", {})

        return self.PROMPT_TEMPLATE.format(
            allowed_domains=allowed_domains_str,
            allowed_categories=allowed_cats_str,
            original_text=text,
            normalized_text=norm_text,
            language=lang,
            signals=json.dumps(signals),
            tfidf_candidates=json.dumps(tfidf_candidates or []),
            embedding_candidates=json.dumps(embedding_candidates or []),
        )

    def _parse_llm_response(self, raw_text: str) -> Dict[str, Any]:
        """Safely parse JSON from LLM response, handling markdown fences and malformed output."""
        cleaned = raw_text.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON object from mixed text
            match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Validate required fields
        if not isinstance(parsed, dict) or "status" not in parsed:
            return None

        return parsed

    def _validate_and_build_result(
        self,
        parsed: Dict[str, Any],
        original_text: str = "",
        tfidf_candidates: Optional[List[Dict[str, Any]]] = None,
        embedding_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Validate LLM output against taxonomy and apply external anti-hallucination guardrails."""
        status = parsed.get("status", "UNKNOWN")
        valid_statuses = {"CLASSIFIED", "AMBIGUOUS", "UNKNOWN", "OUT_OF_SCOPE", "INSUFFICIENT_INFORMATION"}
        if status not in valid_statuses:
            status = "UNKNOWN"

        domain_code = parsed.get("domain_code")
        category_code = parsed.get("category_code")
        confidence = parsed.get("confidence", 0.75)
        raw_evidence = parsed.get("evidence", [])
        if not isinstance(raw_evidence, list):
            raw_evidence = [str(raw_evidence)]

        # 1. Ensure confidence is a valid float
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        # 2. Strict Taxonomy Validation for CLASSIFIED status
        if status == "CLASSIFIED":
            is_valid = False
            if domain_code and category_code:
                dom_info = self.tm.get_domain(domain_code)
                if dom_info and category_code in self.tm.get_categories(domain_code):
                    is_valid = True

            if not is_valid:
                # LLM predicted code outside allowed taxonomy — degrade safely
                status = "AMBIGUOUS"
                domain_code = None
                category_code = None
                confidence = round(confidence * 0.5, 4)
                raw_evidence.append("LLM predicted invalid taxonomy code; degraded externally to AMBIGUOUS")

        # 3. Confidence Threshold Guardrail (Prevent low-confidence guessing)
        if status == "CLASSIFIED" and confidence < 0.60:
            status = "AMBIGUOUS"
            domain_code = None
            category_code = None
            raw_evidence.append("Low model confidence (< 0.60); degraded externally to AMBIGUOUS")

        # 4. Evidence Grounding Guardrail (Remove fake/hallucinated evidence terms)
        text_lower = (original_text or "").lower()
        verified_evidence = []
        if text_lower:
            for item in raw_evidence:
                item_str = str(item).strip()
                if not item_str:
                    continue
                # Keep evidence if it appears in input text or is a system note
                if any(w in text_lower for w in item_str.lower().split() if len(w) > 2) or "degraded" in item_str.lower():
                    verified_evidence.append(item_str)
        else:
            verified_evidence = raw_evidence

        # For non-CLASSIFIED statuses, clear domain/category
        if status != "CLASSIFIED":
            domain_code = None
            category_code = None

        best_dom = domain_code
        best_cat = category_code
        dept_map = self.tm.get_department_mapping(best_cat) if best_cat else None

        return {
            "status": status,
            "domain": {"code": best_dom, "confidence": confidence},
            "category": {"code": best_cat, "confidence": confidence},
            "administrative_mapping": dept_map or {"type": "UNMAPPED", "department_code": None},
            "overall_confidence": round(confidence, 4),
            "reasoning": "Gemini API taxonomy-constrained classification with external guardrails",
            "evidence": verified_evidence,
            "method": "LLM_GEMINI",
        }

    def classify(
        self,
        text: str,
        understanding: Optional[Any] = None,
        tfidf_candidates: Optional[List[Dict[str, Any]]] = None,
        embedding_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes taxonomy-constrained LLM classification.
        Uses Gemini API when available, falls back to offline mock consensus.
        """
        if self.mock_mode:
            return self._mock_classify(text, tfidf_candidates, embedding_candidates)

        # ---- Real Gemini API Call ----
        try:
            prompt = self.build_prompt(text, understanding, tfidf_candidates, embedding_candidates)
            model = self._get_gemini_model()
            response = model.generate_content(prompt)

            raw_text = response.text
            parsed = self._parse_llm_response(raw_text)

            if parsed is None:
                # LLM returned unparseable output — degrade to mock
                return self._mock_classify(text, tfidf_candidates, embedding_candidates)

            return self._validate_and_build_result(
                parsed,
                original_text=text,
                tfidf_candidates=tfidf_candidates,
                embedding_candidates=embedding_candidates,
            )

        except Exception as e:
            # Any API error (rate limit, network, auth) — degrade gracefully to mock
            return self._mock_classify(text, tfidf_candidates, embedding_candidates)

    def _mock_classify(
        self,
        text: str,
        tfidf_candidates: Optional[List[Dict[str, Any]]] = None,
        embedding_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Deterministic offline mock fallback using candidate evidence consensus."""
        best_cat = None
        best_dom = None
        conf = 0.75
        status = "CLASSIFIED"

        # Check candidates
        candidates = (embedding_candidates or []) + (tfidf_candidates or [])
        if candidates:
            top_cand = candidates[0]
            best_cat = top_cand.get("code")
            if best_cat:
                cat_info = self.tm.get_category(best_cat)
                if cat_info:
                    best_dom = None
                    for d_code, d_data in self.tm.data.get("domains", {}).items():
                        if best_cat in d_data.get("categories", {}):
                            best_dom = d_code
                            break

        # Check max candidate score across all models
        max_score = max([c.get("score", c.get("confidence", 0.0)) for c in candidates]) if candidates else 0.0

        # Out of scope / insufficient / ambiguous check
        if not text or len(text.strip().split()) < 3:
            status = "INSUFFICIENT_INFORMATION"
            best_cat = None
            best_dom = None
            conf = 0.30
        elif max_score < 0.45:
            status = "AMBIGUOUS"
            best_cat = None
            best_dom = None
            conf = round(float(max_score), 4)

        dept_map = self.tm.get_department_mapping(best_cat) if best_cat else None

        return {
            "status": status,
            "domain": {"code": best_dom, "confidence": conf},
            "category": {"code": best_cat, "confidence": conf},
            "administrative_mapping": dept_map or {"type": "UNMAPPED", "department_code": None},
            "overall_confidence": conf,
            "reasoning": "Offline candidate consensus (no API key)",
            "evidence": [c.get("code") for c in candidates if c.get("code")],
            "method": "LLM_FALLBACK",
        }
