"""
Targeted tests for Gemini Semantic Resolver and Taxonomy Fit Checker.
Verifies the 8 requested test cases.
"""

import unittest
import os
import sys
from pathlib import Path

# Ensure root dir is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.agents.civicflow_agent import CivicFlowAgent
from ai.tools.gemini_semantic_resolver import GeminiSemanticResolver
from ai.tools.taxonomy_fit_checker import TaxonomyFitChecker

class TestGeminiSemanticResolver(unittest.TestCase):

    def setUp(self):
        self.agent = CivicFlowAgent()
        self.resolver = GeminiSemanticResolver()
        self.fit_checker = TaxonomyFitChecker()

    def test_case_1_roman_marathi_water_pump_complaint(self):
        """1. Roman Marathi water pump failure -> PUBLIC_WATER_PUMP_FAILURE -> UNKNOWN in taxonomy."""
        input_text = "udyacha sakali mi uthlo samor panaycha pump hota tyatun pani nhi ala ani mala ppump nit karachcha aahe"
        res = self.agent.process_complaint(input_text)
        
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertEqual(res["domain_code"], "WATER_SUPPLY")
        self.assertIsNone(res["category_code"])
        self.assertIn("panaycha pump", [e.lower() for e in res["evidence"]] + [input_text.lower()])

    def test_case_2_marathi_water_leak(self):
        """2. 'पाण्याची पाइपलाइन गळत आहे' -> WTR-LEAK."""
        input_text = "पाण्याची पाइपलाइन गळत आहे"
        res = self.agent.process_complaint(input_text)
        
        self.assertEqual(res["status"], "CLASSIFIED")
        self.assertEqual(res["domain_code"], "WATER_SUPPLY")
        self.assertEqual(res["category_code"], "WTR-LEAK")

    def test_case_3_python_out_of_scope(self):
        """3. 'Python garbage collection is slow' -> OUT_OF_SCOPE via Gemini Semantic Resolver."""
        input_text = "Python garbage collection is slow"
        resolved = self.resolver.resolve_intent(input_text)
        fit = self.fit_checker.check_fit(resolved, original_text=input_text)
        
        self.assertEqual(fit["status"], "OUT_OF_SCOPE")
        self.assertIsNone(fit["domain_code"])
        self.assertIsNone(fit["category_code"])

    def test_case_4_vague_water_complaint(self):
        """4. 'पाण्याची समस्या आहे' -> INSUFFICIENT_INFORMATION or UNKNOWN via Gemini Semantic Resolver."""
        input_text = "पाण्याची समस्या आहे"
        resolved = self.resolver.resolve_intent(input_text)
        fit = self.fit_checker.check_fit(resolved, original_text=input_text)
        
        self.assertIn(fit["status"], ["INSUFFICIENT_INFORMATION", "AMBIGUOUS", "UNKNOWN"])
        self.assertIsNone(fit["category_code"])

    def test_case_5_known_pothole_complaint(self):
        """5. Known road pothole complaint -> RDW-POT."""
        input_text = "There is a massive dangerous pothole on the main road creating severe accidents"
        res = self.agent.process_complaint(input_text)
        
        self.assertEqual(res["status"], "CLASSIFIED")
        self.assertEqual(res["category_code"], "RDW-POT")

    def test_case_6_known_garbage_complaint(self):
        """6. Known garbage collection complaint -> SAN-GARB."""
        input_text = "Garbage vehicle has not arrived for three days and trash is overflowing"
        res = self.agent.process_complaint(input_text)
        
        self.assertEqual(res["status"], "CLASSIFIED")
        self.assertEqual(res["category_code"], "SAN-GARB")

    def test_case_7_malformed_gemini_response(self):
        """7. Malformed Gemini response -> safe fallback, no crash."""
        parsed = self.resolver._parse_json_response("This is not JSON text")
        self.assertIsNone(parsed)
        
        fit = self.fit_checker.check_fit(parsed, original_text="some text")
        self.assertEqual(fit["status"], "UNKNOWN")

    def test_case_8_missing_api_key_safe_fallback(self):
        """8. Missing GEMINI_API_KEY -> safe fallback, no crash."""
        resolver = GeminiSemanticResolver(api_key="")
        res = resolver.resolve_intent("udyacha sakali mi uthlo samor panaycha pump hota tyatun pani nhi ala ani mala ppump nit karachcha aahe")
        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "PUBLIC_WATER_PUMP_FAILURE")

if __name__ == "__main__":
    unittest.main()
