"""
Phase 4 Unit Test Suite for CivicFlow AI Calibration, Reliability & Adversarial Protection.
Run via: py -m unittest discover -s tests -p "test_*.py"
"""

import sys
import unittest
from pathlib import Path

# Ensure root workspace in sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.taxonomy.taxonomy_manager import TaxonomyManager
from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent
from ai.agents.uncertainty_gate import UncertaintyGate
from ai.schemas.dataclasses import ClassificationStatus, MappingType


class TestPhase4AgentReliabilityAndCalibration(unittest.TestCase):
    """Test cases for Agent Safety Rule, threshold trace reporting, OOS gate, and adversarial handling."""

    def setUp(self):
        self.tm = TaxonomyManager()
        self.understanding_agent = UnderstandingAgent()
        self.agent = ClassificationAgent(taxonomy_manager=self.tm)
        self.uncertainty_gate = UncertaintyGate()

    def test_1_confident_tfidf_early_stopping(self):
        """Verify high confidence TF-IDF prediction accepts early without escalation."""
        text = "Garbage vehicle has not arrived for three days in society. Trash is overflowing near main gate."
        u = self.understanding_agent.understand(text)
        res = self.agent.classify(u, debug=True)

        self.assertEqual(res.method, "TFIDF")
        self.assertEqual(res.status, ClassificationStatus.CLASSIFIED.value)
        self.assertEqual(res.semantic_classification.category_code, "SAN-GARB")

        trace = res.secondary_signals[0].get("trace", [])
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["decision"], "ACCEPT")

    def test_2_weak_tfidf_escalates(self):
        """Verify weak TF-IDF confidence or low margin escalates to secondary tool."""
        weak_agent = ClassificationAgent(taxonomy_manager=self.tm, tfidf_threshold=0.99, tfidf_margin=0.50)
        text = "Garbage vehicle missing."
        u = self.understanding_agent.understand(text)
        res = weak_agent.classify(u, debug=True)

        trace = res.secondary_signals[0].get("trace", [])
        self.assertGreaterEqual(len(trace), 2)
        self.assertEqual(trace[0]["decision"], "ESCALATE")

    def test_3_adversarial_oos_complaint_handling(self):
        """Verify non-civic adversarial query with municipal keywords is correctly flagged OUT_OF_SCOPE."""
        text = "My smartphone garbage collection app is crashing on Android."
        status, reason = self.uncertainty_gate.evaluate(text, text.lower())
        self.assertEqual(status, ClassificationStatus.OUT_OF_SCOPE.value)

        u = self.understanding_agent.understand(text)
        res = self.agent.classify(u)
        self.assertEqual(res.status, ClassificationStatus.OUT_OF_SCOPE.value)
        self.assertIsNone(res.semantic_classification.category_code)

    def test_4_insufficient_information_handling(self):
        """Verify short/vague text is flagged INSUFFICIENT_INFORMATION."""
        text = "Please solve."
        u = self.understanding_agent.understand(text)
        res = self.agent.classify(u)
        self.assertEqual(res.status, ClassificationStatus.INSUFFICIENT_INFORMATION.value)

    def test_5_ambiguous_complaint_handling(self):
        """Verify vague municipal query lacking intent is flagged AMBIGUOUS or INSUFFICIENT_INFORMATION."""
        text = "There is some big problem near my house."
        u = self.understanding_agent.understand(text)
        res = self.agent.classify(u)
        self.assertIn(res.status, [ClassificationStatus.INSUFFICIENT_INFORMATION.value, ClassificationStatus.AMBIGUOUS.value])

    def test_6_trace_contains_scores_and_thresholds(self):
        """Verify trace entries record raw_score, confidence, margin, threshold, decision, reason, final_selection."""
        text = "Main water supply pipeline burst near bus stop."
        u = self.understanding_agent.understand(text)
        res = self.agent.classify(u, debug=True)

        trace = res.secondary_signals[0].get("trace", [])
        self.assertGreater(len(trace), 0)
        entry = trace[0]

        self.assertIn("tool", entry)
        self.assertIn("confidence", entry)
        self.assertIn("threshold", entry)
        self.assertIn("decision", entry)
        self.assertIn("reason", entry)
        self.assertIn("final_selection", entry)

    def test_7_no_challenge_data_used_for_threshold_tuning(self):
        """Verify tuning script uses validation.csv and not challenge_test.csv."""
        from ai.evaluation.tune_thresholds import ThresholdTuner
        tuner = ThresholdTuner()
        val_csv = Path(__file__).parent.parent / "ai" / "data" / "processed" / "validation.csv"
        self.assertTrue(val_csv.exists())
        self.assertNotIn("challenge_test.csv", str(val_csv))

    def test_8_false_confident_classification_protection(self):
        """Verify non-municipal query is never forced into a normal 34-category classification label."""
        text = "Order laptop online from website store."
        u = self.understanding_agent.understand(text)
        res = self.agent.classify(u)
        self.assertNotEqual(res.status, ClassificationStatus.CLASSIFIED.value)
        self.assertIsNone(res.semantic_classification.category_code)


if __name__ == "__main__":
    unittest.main()
