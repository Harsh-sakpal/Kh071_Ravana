"""
Taxonomy Fit Checker for Gemini Semantic Resolver Output.

Compares Gemini's extracted semantic intent against taxonomy.json.
Taxonomy.json remains the SINGLE SOURCE OF TRUTH.
"""

from typing import Dict, Any, Optional, List
from ai.taxonomy.taxonomy_manager import TaxonomyManager
from ai.schemas.dataclasses import ClassificationStatus, MappingType

NON_WATER_SUBSTANCES = {"oil", "gas", "petroleum", "chemical", "fuel", "diesel", "petrol"}
NON_WATER_INTENTS = {"OIL_PIPE_FAILURE", "GAS_PIPE_LEAK", "CHEMICAL_PIPELINE_LEAK", "FUEL_SPILL"}


class TaxonomyFitChecker:
    """Evaluates whether a resolved semantic intent fits into the current taxonomy."""

    def __init__(self, taxonomy_manager: Optional[TaxonomyManager] = None):
        self.tm = taxonomy_manager or TaxonomyManager()

    def check_fit(self, intent_data: Dict[str, Any], original_text: str = "") -> Dict[str, Any]:
        """
        Takes output of GeminiSemanticResolver and evaluates against taxonomy.json.

        Decisions:
        - Non-water pipe leak (oil, gas, chemical) -> OUT_OF_SCOPE (NEVER WTR-LEAK)
        - Non-civic query -> OUT_OF_SCOPE
        - Vague / insufficient intent -> INSUFFICIENT_INFORMATION
        - Public water pump failure -> UNKNOWN (domain WATER_SUPPLY, category None)
        - Valid water pipe leak -> CLASSIFIED (WTR-LEAK)
        """
        if not intent_data or not isinstance(intent_data, dict):
            return {
                "status": ClassificationStatus.UNKNOWN.value,
                "domain_code": None,
                "category_code": None,
                "confidence": 0.0,
                "evidence": [],
                "explanation": "Unable to resolve semantic intent.",
            }

        domain_code = intent_data.get("domain_code")
        intent = intent_data.get("intent", "")
        problem = intent_data.get("problem", "")
        requested_action = intent_data.get("requested_action", "")
        obj = (intent_data.get("object") or "").lower()
        substance = (intent_data.get("material_or_substance") or "").lower()
        raw_evidence = intent_data.get("evidence", [])
        if not isinstance(raw_evidence, list):
            raw_evidence = [str(raw_evidence)]

        # Evidence grounding check
        text_lower = (original_text or "").lower()
        evidence = []
        if text_lower:
            for ev in raw_evidence:
                ev_str = str(ev).strip()
                if not ev_str:
                    continue
                if ev_str.lower() in text_lower or any(w in text_lower for w in ev_str.lower().split() if len(w) > 2):
                    evidence.append(ev_str)
        if not evidence:
            evidence = raw_evidence[:3]

        # -------------------------------------------------------------------
        # 1. NON-WATER PIPELINE LEAK / SUBSTANCE GUARD (CRITICAL RULE)
        # -------------------------------------------------------------------
        import re
        is_non_water_substance = substance in NON_WATER_SUBSTANCES
        is_non_water_intent = intent in NON_WATER_INTENTS
        has_non_water_keywords = any(
            kw in text_lower
            for kw in ["oil pipe", "oil pipeline", "oil transport", "gas pipe", "gas pipeline", "chemical pipe", "tel nighayla", "तेल", "ऑईल", "गॅस", "तेलाच्या"]
        ) or bool(re.search(r"\b(oil|oils|gas|gases|petroleum|chemical|chemicals|diesel|fuel|kerosene)\b", text_lower))

        if is_non_water_substance or is_non_water_intent or has_non_water_keywords:
            return {
                "status": ClassificationStatus.OUT_OF_SCOPE.value,
                "domain_code": None,
                "category_code": None,
                "confidence": 0.95,
                "evidence": evidence,
                "explanation": f"Pipe failure involving non-water substance ({substance or 'oil/gas'}) is outside municipal water supply scope.",
            }

        # 2. OUT OF SCOPE check
        if domain_code == "OUT_OF_SCOPE" or intent == "SOFTWARE_PROGRAMMING_QUERY":
            return {
                "status": ClassificationStatus.OUT_OF_SCOPE.value,
                "domain_code": None,
                "category_code": None,
                "confidence": 0.95,
                "evidence": evidence,
                "explanation": "Complaint is unrelated to municipal civic duties.",
            }

        # 3. INSUFFICIENT INFORMATION check
        if intent in ("VAGUE_WATER_ISSUE", "INSUFFICIENT_INFORMATION") or "without specific details" in problem.lower():
            return {
                "status": ClassificationStatus.INSUFFICIENT_INFORMATION.value,
                "domain_code": domain_code if self.tm.get_domain(domain_code) else None,
                "category_code": None,
                "confidence": 0.90,
                "evidence": evidence,
                "explanation": "Complaint is too vague or lacks necessary details.",
            }

        # 4. Check Domain Validity in taxonomy.json
        dom_info = self.tm.get_domain(domain_code) if domain_code else None
        if not dom_info:
            return {
                "status": ClassificationStatus.UNKNOWN.value,
                "domain_code": None,
                "category_code": None,
                "confidence": 0.50,
                "evidence": evidence,
                "explanation": f"Domain {domain_code} does not exist in taxonomy.",
            }

        # 5. Category Matching Logic against taxonomy.json categories
        matched_cat_code = self._match_intent_to_category(domain_code, intent, problem, original_text, substance, obj)

        if matched_cat_code:
            dept_map = self.tm.get_department_mapping(matched_cat_code)
            return {
                "status": ClassificationStatus.CLASSIFIED.value,
                "domain_code": domain_code,
                "category_code": matched_cat_code,
                "confidence": 0.92,
                "administrative_mapping": dept_map or {"type": MappingType.DIRECT.value, "department_code": None},
                "evidence": evidence,
                "explanation": f"Classified based on semantic match to intent '{intent}'.",
            }
        else:
            # NO matching category in taxonomy.json!
            # Example: PUBLIC_WATER_PUMP_FAILURE -> returns UNKNOWN with domain_code=WATER_SUPPLY.
            return {
                "status": ClassificationStatus.UNKNOWN.value,
                "domain_code": domain_code,
                "category_code": None,
                "confidence": 0.90,
                "administrative_mapping": {"type": MappingType.UNMAPPED.value, "department_code": None},
                "evidence": evidence,
                "explanation": f"The complaint describes {problem.lower() if problem else intent}, but no matching taxonomy category exists.",
            }

    def _match_intent_to_category(
        self, domain_code: str, intent: str, problem: str, original_text: str, substance: str, obj: str
    ) -> Optional[str]:
        """Strict semantic matching against category descriptions and example_intents in taxonomy.json."""
        categories = self.tm.get_categories(domain_code)
        text_lower = (original_text or "").lower()

        # Explicit non-matches / intent guards:
        # Public water pump failure -> NOT WTR-LEAK or WTR-PRES!
        if intent == "PUBLIC_WATER_PUMP_FAILURE" or obj == "pump" or "pump" in text_lower or "पंप" in original_text:
            return None

        # Water Main Leak: Only match if substance is water (or unspecified water context)
        is_water_context = substance in ("", "water", "none") or any(
            w in text_lower for w in ["water", "pani", "panyachi", "पाणी", "पाण्याची"]
        )
        if (intent == "WATER_MAIN_LEAK" or "pipeline leak" in problem.lower() or "गळत" in original_text) and is_water_context:
            if "WTR-LEAK" in categories:
                return "WTR-LEAK"

        if intent == "POTHOLE_REPAIR" or "pothole" in text_lower or "khadda" in text_lower or "खड्डा" in original_text:
            if "RDW-POT" in categories:
                return "RDW-POT"

        if intent in ("EXPOSED_WIRING_HAZARD", "ELECTRICAL_LEAKAGE") or "electricity" in text_lower or "vidyut" in text_lower or "current leak" in text_lower:
            if "ELE-POLE" in categories:
                return "ELE-POLE"

        if intent == "UNCOLLECTED_GARBAGE" or "garbage" in text_lower or "kachra" in text_lower or "कचरा" in original_text:
            if "SAN-GARB" in categories:
                return "SAN-GARB"

        # General check against category example_intents and descriptions
        for cat_code, cat_info in categories.items():
            example_intents = cat_info.get("example_intents", [])
            cat_desc = cat_info.get("description", "").lower()

            for ex in example_intents:
                if intent.lower().replace("_", " ") in ex.lower():
                    return cat_code

        return None
