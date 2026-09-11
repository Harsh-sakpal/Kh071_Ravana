import unittest
from ai.agents.civicflow_agent import CivicFlowAgent
from ai.schemas.dataclasses import ClassificationStatus

class TestCivicFlowAgent(unittest.TestCase):
    def setUp(self):
        self.agent = CivicFlowAgent()

    def test_english_classified(self):
        res = self.agent.process_complaint("Pothole on main road", debug=False)
        self.assertEqual(res["status"], ClassificationStatus.CLASSIFIED.value)
        self.assertEqual(res["category_code"], "RDW-POT")
        self.assertNotIn("trace", res)

    def test_marathi_classified(self):
        res = self.agent.process_complaint("रस्त्यावर खड्डे आहेत", debug=True)
        self.assertEqual(res["status"], ClassificationStatus.CLASSIFIED.value)
        self.assertIn("trace", res)
        self.assertIn(res["trace"]["language"], ["MR", "MARATHI"])

    def test_roman_marathi_classified(self):
        res = self.agent.process_complaint("rasta kharab aahe kachra ahe", debug=True)
        self.assertEqual(res["status"], ClassificationStatus.CLASSIFIED.value)
        self.assertIn(res["trace"]["language"], ["HINGLISH", "MR", "MARATHI", "MIXED"])
        
    def test_hinglish_classified(self):
        res = self.agent.process_complaint("mera road kharab hai", debug=True)
        self.assertEqual(res["status"], ClassificationStatus.CLASSIFIED.value)
        self.assertIn(res["trace"]["language"], ["HINGLISH", "MIXED", "EN"])

    def test_hard_negative_oos(self):
        res = self.agent.process_complaint("My garbage collection app keeps crashing on Android.", debug=False)
        self.assertEqual(res["status"], ClassificationStatus.OUT_OF_SCOPE.value)
        
    def test_ambiguous(self):
        original = self.agent.classification_agent.classify
        def return_ambiguous(*args, **kwargs):
            from ai.schemas.dataclasses import ClassificationOutput, SemanticClassification, AdministrativeMapping
            return ClassificationOutput(
                status=ClassificationStatus.AMBIGUOUS.value,
                semantic_classification=SemanticClassification(),
                administrative_mapping=AdministrativeMapping(),
                overall_confidence=0.3,
                method="MOCK",
                evidence=["Conflicting signals detected"]
            )
        self.agent.classification_agent.classify = return_ambiguous
        
        try:
            res = self.agent.process_complaint("Everything is broken, the water, the roads, the lights, the garbage", debug=False)
            self.assertEqual(res["status"], ClassificationStatus.AMBIGUOUS.value)
        finally:
            self.agent.classification_agent.classify = original
        res1 = self.agent.process_complaint("", debug=False)
        self.assertEqual(res1["status"], ClassificationStatus.INSUFFICIENT_INFORMATION.value)
        
        res2 = self.agent.process_complaint(None, debug=False)
        self.assertEqual(res2["status"], ClassificationStatus.INSUFFICIENT_INFORMATION.value)
        
        res3 = self.agent.process_complaint("   \n ", debug=False)
        self.assertEqual(res3["status"], ClassificationStatus.INSUFFICIENT_INFORMATION.value)

    def test_simulated_tfidf_failure(self):
        # Force tfidf to raise exception
        original = self.agent.classification_agent.tfidf_tool.classify
        def raise_err(*args, **kwargs):
            raise RuntimeError("TFIDF Crash Simulation")
        self.agent.classification_agent.tfidf_tool.classify = raise_err
        
        try:
            res = self.agent.process_complaint("Pothole on main road", debug=True)
            # Classification Agent catches tool errors and routes to fallback/embedding.
            # Even if it completely blows up, the CivicFlowAgent catches it.
            self.assertIn(res["status"], [ClassificationStatus.UNKNOWN.value, ClassificationStatus.CLASSIFIED.value, ClassificationStatus.AMBIGUOUS.value])
            # Check zero unhandled exceptions
        finally:
            self.agent.classification_agent.tfidf_tool.classify = original

    def test_simulated_llm_failure(self):
        # Force LLM to raise exception
        original = self.agent.classification_agent.llm_tool.classify
        def raise_err(*args, **kwargs):
            raise RuntimeError("LLM Crash Simulation")
        self.agent.classification_agent.llm_tool.classify = raise_err
        
        try:
            # Send something that bypasses tfidf threshold
            res = self.agent.process_complaint("A highly unusual issue that needs LLM resolution but LLM crashes", debug=True)
            self.assertIn(res["status"], [ClassificationStatus.AMBIGUOUS.value, ClassificationStatus.UNKNOWN.value, ClassificationStatus.CLASSIFIED.value])
        finally:
            self.agent.classification_agent.llm_tool.classify = original
            
    def test_taxonomy_validation(self):
        # Force ClassificationAgent to return invalid taxonomy
        original = self.agent.classification_agent.classify
        def return_invalid(*args, **kwargs):
            from ai.schemas.dataclasses import ClassificationOutput, SemanticClassification, AdministrativeMapping
            return ClassificationOutput(
                status=ClassificationStatus.CLASSIFIED.value,
                semantic_classification=SemanticClassification(domain_code="FAKE_DOMAIN", category_code="FAKE_CAT"),
                administrative_mapping=AdministrativeMapping(),
                overall_confidence=0.99,
                method="MOCK",
                evidence=["Mock"]
            )
        self.agent.classification_agent.classify = return_invalid
        
        try:
            res = self.agent.process_complaint("Pothole", debug=True)
            self.assertEqual(res["status"], ClassificationStatus.UNKNOWN.value) # intercepted and fixed
            self.assertIsNone(res["domain_code"])
            self.assertIsNone(res["category_code"])
        finally:
            self.agent.classification_agent.classify = original
            
    def test_no_cot_in_production(self):
        res = self.agent.process_complaint("Pothole on main road", debug=False)
        explanation = res.get("explanation", "").lower()
        self.assertNotIn("thought", explanation)
        self.assertNotIn("chain", explanation)
        self.assertNotIn("{", explanation)
        
if __name__ == '__main__':
    unittest.main()
