"""
CivicFlow AI - Semantic Boundary Repair Regression Test Suite
===============================================================

Verifies entity/substance disambiguation, taxonomy fit guardrails,
out-of-scope non-water pipeline detection, and non-hallucinated explanations.
"""

import unittest
from ai.agents.civicflow_agent import CivicFlowAgent
from ai.agents.classification_agent import ClassificationAgent
from ai.agents.understanding_agent import UnderstandingAgent
from ai.tools.taxonomy_fit_checker import TaxonomyFitChecker
from ai.tools.gemini_semantic_resolver import GeminiSemanticResolver


class TestSemanticBoundaryRepair(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.agent = CivicFlowAgent()
        cls.classifier = ClassificationAgent()
        cls.understanding = UnderstandingAgent()
        cls.fit_checker = TaxonomyFitChecker()
        cls.resolver = GeminiSemanticResolver()

    # -----------------------------------------------------------------------
    # A. KNOWN WATER CASES (Must be classified as WTR-LEAK)
    # -----------------------------------------------------------------------
    def test_water_case_1_english(self):
        res = self.agent.process_complaint("water is leaking from pipe")
        self.assertEqual(res["status"], "CLASSIFIED")
        self.assertEqual(res["category_code"], "WTR-LEAK")

    def test_water_case_2_devanagari_marathi(self):
        res = self.agent.process_complaint("पाण्याची पाइपलाइन गळत आहे")
        self.assertEqual(res["status"], "CLASSIFIED")
        self.assertEqual(res["category_code"], "WTR-LEAK")

    def test_water_case_3_roman_marathi(self):
        res = self.agent.process_complaint("panyachi pipe leak zali aahe")
        self.assertEqual(res["status"], "CLASSIFIED")
        self.assertEqual(res["category_code"], "WTR-LEAK")

    # -----------------------------------------------------------------------
    # B. NON-WATER PIPE CASES (MUST NEVER BE WTR-LEAK!)
    # -----------------------------------------------------------------------
    def test_non_water_case_4_oil_pipe_english(self):
        res = self.agent.process_complaint("oil is leaking from the pipe")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")
        self.assertIn(res["status"], ["OUT_OF_SCOPE", "UNKNOWN"])

    def test_non_water_case_5_oil_pipe_marathi(self):
        res = self.agent.process_complaint("तेलाच्या पाइपमधून तेल बाहेर येत आहे")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")
        self.assertIn(res["status"], ["OUT_OF_SCOPE", "UNKNOWN"])

    def test_non_water_case_6_roman_marathi_oil_pipe_demonstrated_failure(self):
        """Primary reported failure case."""
        complaint = "AAj aamchya samorcha pipe phutla ani tyatun tel nighayla lagle mag samajla ki to oil pipe hota"
        res = self.agent.process_complaint(complaint)
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")
        self.assertIn(res["status"], ["OUT_OF_SCOPE", "UNKNOWN"])

    def test_non_water_case_7_gas_pipe_english(self):
        res = self.agent.process_complaint("gas pipe is leaking")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")
        self.assertIn(res["status"], ["OUT_OF_SCOPE", "UNKNOWN"])

    def test_non_water_case_8_gas_pipe_roman_marathi(self):
        res = self.agent.process_complaint("gas pipeline madhun gas leak hot aahe")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")
        self.assertIn(res["status"], ["OUT_OF_SCOPE", "UNKNOWN"])

    # -----------------------------------------------------------------------
    # C. CLOSE WATER BOUNDARIES
    # -----------------------------------------------------------------------
    def test_boundary_case_9_public_water_pump(self):
        """Public water pump failure has no category in taxonomy -> UNKNOWN (domain WATER_SUPPLY)."""
        res = self.agent.process_complaint("public water pump is broken")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")
        self.assertEqual(res["status"], "UNKNOWN")

    def test_boundary_case_10_water_pressure(self):
        res = self.agent.process_complaint("water pressure is very low")
        self.assertEqual(res["status"], "CLASSIFIED")
        self.assertEqual(res["category_code"], "WTR-PRES")

    def test_boundary_case_11_water_quality(self):
        res = self.agent.process_complaint("dirty water is coming from tap")
        self.assertEqual(res["status"], "CLASSIFIED")
        self.assertEqual(res["category_code"], "WTR-QUAL")

    def test_boundary_case_12_sewage_pipe(self):
        res = self.agent.process_complaint("sewage pipe is blocked and overflowing")
        self.assertEqual(res["status"], "CLASSIFIED")
        self.assertEqual(res["category_code"], "SEW-BLOCK")

    # -----------------------------------------------------------------------
    # D. ADVERSARIAL KEYWORD TESTS
    # -----------------------------------------------------------------------
    def test_adversarial_13_oil_pipe_with_nearby_water(self):
        res = self.agent.process_complaint("oil pipe leaking but there is water nearby")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")
        self.assertIn(res["status"], ["OUT_OF_SCOPE", "UNKNOWN"])

    def test_adversarial_14_water_word_in_gas_pipe_sentence(self):
        res = self.agent.process_complaint("water mentioned in a sentence but actual object is gas pipe")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")
        self.assertIn(res["status"], ["OUT_OF_SCOPE", "UNKNOWN"])

    def test_adversarial_15_pipe_burst_after_oil_transport(self):
        res = self.agent.process_complaint("pipe burst after oil transport")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")

    def test_adversarial_16_python_library_oos(self):
        res = self.agent.process_complaint("Python pipe library water example code crash")
        self.assertEqual(res["status"], "OUT_OF_SCOPE")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")

    def test_adversarial_17_oil_leak_game_oos(self):
        res = self.agent.process_complaint("oil leak in a game minecraft simulation level")
        self.assertEqual(res["status"], "OUT_OF_SCOPE")

    # -----------------------------------------------------------------------
    # E. ELECTRICITY / POWER LEAKAGE TESTS
    # -----------------------------------------------------------------------
    def test_electricity_leak_19_english(self):
        res = self.agent.process_complaint("electricity is leaking in our area pls fix the system")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")

    def test_electricity_leak_20_marathi(self):
        res = self.agent.process_complaint("rastyavaril vidyut khambyavarun current leak hot aahe")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")

    # -----------------------------------------------------------------------
    # F. MULTIPLE-ISSUE TEST
    # -----------------------------------------------------------------------
    def test_multiple_issue_21_oil_pipe_and_street_light(self):
        res = self.agent.process_complaint("Oil pipe burst and nearby streetlight is also not working.")
        self.assertNotEqual(res.get("category_code"), "WTR-LEAK")


if __name__ == "__main__":
    unittest.main()
