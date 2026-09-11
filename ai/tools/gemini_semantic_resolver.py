"""
Gemini Semantic Intent Resolver for CivicFlow AI.

Understands the underlying semantic intent of a complaint (especially multilingual
and Roman Marathi complaints) BEFORE taxonomy matching.

Strict Rules:
- Does NOT invent CivicFlow category codes.
- No Chain-of-Thought / hidden reasoning.
- Extracts material_or_substance and object for entity-first disambiguation.
- Safely handles missing GEMINI_API_KEY, timeouts, network failures, or malformed JSON.
"""

import os
import json
import re
from typing import Dict, Any, Optional
from ai.taxonomy.taxonomy_manager import TaxonomyManager

PROMPT_TEMPLATE = """You are a multilingual semantic intent extractor for civic municipal complaints.
Analyze the user's complaint and determine its semantic intent.

Input Complaint: "{text}"

INSTRUCTIONS:
1. Identify language (EN, MR, ROMAN_MARATHI, HINGLISH, OTHER).
2. Identify the specific object/infrastructure involved (e.g. pipeline, pump, road, streetlight, garbage_bin).
3. Identify the leaking substance or material if applicable (e.g. water, sewage, oil, gas, chemical, petroleum, unknown).
4. Identify the broad municipal domain code (e.g. WATER_SUPPLY, SOLID_WASTE, ROADS_FOOTPATHS, STREET_LIGHTING, SEWERAGE_DRAINAGE, STORM_WATER, GARDENS_TREES, PUBLIC_HEALTH, BUILDING_CONSTRUCTION, ENCROACHMENT, TRAFFIC_PARKING, PROPERTY_TAX, LICENSES_PERMISSIONS, ENVIRONMENT_POLLUTION, FIRE_DISASTER_SAFETY). If not civic or non-municipal infrastructure (e.g., oil pipeline, gas line, software, personal shopping), set domain_code to "OUT_OF_SCOPE".
5. Describe the intent as an uppercase snake_case phrase representing the specific problem (e.g. OIL_PIPE_FAILURE, GAS_PIPE_LEAK, PUBLIC_WATER_PUMP_FAILURE, WATER_MAIN_LEAK, POTHOLE_REPAIR, UNCOLLECTED_GARBAGE).
6. Summarize the core problem in 1 plain text sentence grounded ONLY in the text.
7. Summarize the requested action in 1 plain text sentence.
8. Extract 1 to 4 direct evidence phrases/words from the original text.

CRITICAL RULES:
- DO NOT invent or predict CivicFlow category codes (like WTR-LEAK or RDW-POT).
- Return ONLY a raw valid JSON object. No markdown, no code fences, no reasoning text.

JSON Schema:
{{
  "language": "EN|MR|ROMAN_MARATHI|HINGLISH|OTHER",
  "object": "<object_type>",
  "material_or_substance": "<substance_name>",
  "domain_code": "<DOMAIN_CODE>",
  "intent": "<UPPERCASE_INTENT_PHRASE>",
  "problem": "<Short description of the problem>",
  "requested_action": "<Short description of requested action>",
  "evidence": ["<phrase1>", "<phrase2>"]
}}
"""


class GeminiSemanticResolver:
    """Component that calls Gemini API or offline resolver to extract semantic intent."""

    def __init__(self, api_key: Optional[str] = None, taxonomy_manager: Optional[TaxonomyManager] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.tm = taxonomy_manager or TaxonomyManager()
        self._gemini_model = None

    def _get_model(self):
        if self._gemini_model is None and self.api_key:
            try:
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
            except Exception:
                self._gemini_model = None
        return self._gemini_model

    def resolve_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """Calls Gemini to get semantic intent or falls back to deterministic offline resolver."""
        if not text or not isinstance(text, str) or not text.strip():
            return None

        model = self._get_model()
        if not model:
            return self._mock_semantic_resolution(text)

        try:
            prompt = PROMPT_TEMPLATE.format(text=text)
            response = model.generate_content(prompt)
            raw_text = response.text
            res = self._parse_json_response(raw_text)
            if res:
                return res
            return self._mock_semantic_resolution(text)
        except Exception:
            return self._mock_semantic_resolution(text)

    def _parse_json_response(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON response from LLM."""
        if not raw_text:
            return None
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        if not isinstance(parsed, dict) or "domain_code" not in parsed:
            return None

        return parsed

    def _mock_semantic_resolution(self, text: str) -> Optional[Dict[str, Any]]:
        """Deterministic offline semantic resolver for testing when GEMINI_API_KEY is not set."""
        text_lower = text.lower()

        # 0. Electricity / Electrical current leakage
        if any(w in text_lower for w in ["electricity", "electrical", "vidyut", "करंट", "current"]):
            return {
                "language": "ROMAN_MARATHI" if "khambyavarun" in text_lower else ("MR" if "विद्युत" in text_lower else "EN"),
                "object": "light pole",
                "material_or_substance": "electricity",
                "domain_code": "STREET_LIGHTING",
                "intent": "EXPOSED_WIRING_HAZARD",
                "problem": "Electrical current or electricity leaking from light pole or wiring.",
                "requested_action": "Fix electricity leakage hazard.",
                "evidence": ["electricity", "leaking"] if "electricity" in text_lower else ["vidyut", "current leak"],
            }

        # 1. Oil pipeline failure (Roman Marathi, Marathi, English)
        if any(w in text_lower for w in ["oil", "tel", "तेल", "तेलाच्या"]) and any(w in text_lower for w in ["pipe", "pipeline", "पाइप", "पाइपलाइन"]):
            ev = []
            if "pipe phutla" in text_lower:
                ev.append("pipe phutla")
            if "tel nighayla lagle" in text_lower or "tel nighayla" in text_lower:
                ev.append("tel nighayla lagle")
            if "oil pipe" in text_lower:
                ev.append("oil pipe")
            if "तेल" in text_lower:
                ev.append("तेलाच्या पाइपमधून")
            if not ev:
                ev = ["oil pipe", "leak"]

            return {
                "language": "ROMAN_MARATHI" if "phutla" in text_lower else ("MR" if "तेल" in text_lower else "EN"),
                "object": "pipeline",
                "material_or_substance": "oil",
                "domain_code": "OUT_OF_SCOPE",
                "intent": "OIL_PIPE_FAILURE",
                "problem": "Oil is leaking from a burst oil pipeline.",
                "requested_action": "None",
                "evidence": ev,
            }

        # 2. Gas pipeline failure (Roman Marathi, Marathi, English)
        if any(w in text_lower for w in ["gas", "गॅस"]) and any(w in text_lower for w in ["pipe", "pipeline", "पाइप", "पाइपलाइन"]):
            return {
                "language": "ROMAN_MARATHI" if "madhun" in text_lower else ("MR" if "गॅस" in text_lower else "EN"),
                "object": "pipeline",
                "material_or_substance": "gas",
                "domain_code": "OUT_OF_SCOPE",
                "intent": "GAS_PIPE_LEAK",
                "problem": "Gas is leaking from a gas pipeline.",
                "requested_action": "None",
                "evidence": ["gas pipeline", "gas leak"],
            }

        # 3. Chemical / Industrial pipeline failure
        if any(w in text_lower for w in ["chemical", "industrial", "toxic"]) and any(w in text_lower for w in ["pipe", "pipeline", "leak"]):
            return {
                "language": "EN",
                "object": "pipeline",
                "material_or_substance": "chemical",
                "domain_code": "OUT_OF_SCOPE",
                "intent": "CHEMICAL_PIPELINE_LEAK",
                "problem": "Chemical is leaking from an industrial pipeline.",
                "requested_action": "None",
                "evidence": ["chemical pipeline", "leak"],
            }

        # 4. Public water pump failure (Roman Marathi, Marathi, English)
        if any(w in text_lower for w in ["pump", "पंप"]) and any(w in text_lower for w in ["panaycha", "pani", "पाणी", "पाण्याचा", "water"]):
            ev = [w for w in ["panaycha pump", "pani nhi ala", "pump nit karachcha", "पाण्याचा पंप", "public water pump"] if any(k in text_lower for k in w.split())]
            return {
                "language": "ROMAN_MARATHI" if "panaycha" in text_lower else ("MR" if "पाणी" in text_lower else "EN"),
                "object": "pump",
                "material_or_substance": "water",
                "domain_code": "WATER_SUPPLY",
                "intent": "PUBLIC_WATER_PUMP_FAILURE",
                "problem": "Water is not coming from a public water pump.",
                "requested_action": "Repair the pump.",
                "evidence": ev or ["pump", "water"],
            }

        # 5. Water pipeline leak (Roman Marathi, Marathi, English)
        if any(w in text_lower for w in ["leaking", "leak", "गळत", "गळती", "burst"]) and any(w in text_lower for w in ["water", "pani", "panyachi", "पाणी", "पाण्याची"]):
            if not any(w in text_lower for w in ["oil", "gas", "tel", "chemical"]):
                return {
                    "language": "MR" if "पाइपलाइन" in text_lower or "गळत" in text_lower else ("ROMAN_MARATHI" if "panyachi" in text_lower else "EN"),
                    "object": "pipeline",
                    "material_or_substance": "water",
                    "domain_code": "WATER_SUPPLY",
                    "intent": "WATER_MAIN_LEAK",
                    "problem": "Water main pipeline is leaking.",
                    "requested_action": "Repair water pipeline leak.",
                    "evidence": ["पाण्याची पाइपलाइन", "गळत आहे"] if "पाण्याची" in text_lower else (["panyachi pipe", "leak zali aahe"] if "panyachi" in text_lower else ["water is leaking", "pipe"]),
                }

        # 6. Pothole / Road damage pattern (Marathi & English)
        if any(w in text_lower for w in ["खड्डे", "खड्ड्यात", "pothole", "potholes", "rasta"]):
            return {
                "language": "MR" if "खड्डे" in text_lower or "रस्त्यावर" in text_lower else "EN",
                "object": "road",
                "material_or_substance": "asphalt",
                "domain_code": "ROADS_FOOTPATHS",
                "intent": "POTHOLE_REPAIR",
                "problem": "Road has potholes.",
                "requested_action": "Repair road potholes.",
                "evidence": ["खड्डे पडले आहेत", "मुख्य रस्त्यावर"] if "खड्डे" in text_lower else ["potholes", "road"],
            }

        # 7. Python / Game / Non-civic pattern
        if any(w in text_lower for w in ["python", "garbage collection", "code", "programming", "software", "game"]):
            return {
                "language": "EN",
                "object": "software",
                "material_or_substance": "none",
                "domain_code": "OUT_OF_SCOPE",
                "intent": "SOFTWARE_PROGRAMMING_QUERY",
                "problem": "Query is regarding software programming or gaming rather than municipal civic duties.",
                "requested_action": "None",
                "evidence": [w for w in ["python", "garbage collection", "game", "code"] if w in text_lower],
            }

        # 8. Vague water pattern
        if any(w in text_lower for w in ["pani", "पाणी", "water"]) and len(text.strip().split()) <= 4 and not any(w in text_lower for w in ["pump", "leak", "pipe", "gatar", "galt"]):
            return {
                "language": "MR" if "पाण्याची" in text_lower else "EN",
                "object": "water_supply",
                "material_or_substance": "water",
                "domain_code": "WATER_SUPPLY",
                "intent": "VAGUE_WATER_ISSUE",
                "problem": "Water issue mentioned without specific details.",
                "requested_action": "Provide specific details.",
                "evidence": ["पाण्याची समस्या"] if "पाण्याची" in text_lower else ["water issue"],
            }

        return None
