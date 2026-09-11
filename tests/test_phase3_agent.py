"""
Phase 3 Unit Test Suite for CivicFlow AI Agentic Classification & Sentence Embeddings.
Run via: py -m unittest discover -s tests -p "test_*.py"
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure root workspace in sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.taxonomy.taxonomy_manager import TaxonomyManager
from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent
from ai.tools.tfidf_tool import TfidfClassificationTool
from ai.tools.llm_tool import LlmFallbackTool
from ai.schemas.dataclasses import ClassificationStatus, MappingType


class TestPhase3AgenticClassification(unittest.TestCase):
    """Test cases for embedding tool, agentic tool escalation policy, trace, and out-of-scope handling."""

    def setUp(self):
        self.tm = TaxonomyManager()
        self.understanding_agent = UnderstandingAgent()
        self.classification_agent = ClassificationAgent(taxonomy_manager=self.tm)

    def test_1_high_confidence_tfidf_avoids_escalation(self):
        """Verify high confidence TF-IDF prediction accepts immediately without calling secondary tools."""
        text = "Garbage vehicle has not arrived for three days in society. Trash is overflowing near main gate."
        understanding = self.understanding_agent.understand(text)
        res = self.classification_agent.classify(understanding, debug=True)

        self.assertEqual(res.method, "TFIDF")
        self.assertEqual(res.status, ClassificationStatus.CLASSIFIED.value)
        self.assertEqual(res.semantic_classification.domain_code, "SOLID_WASTE")
        self.assertEqual(res.semantic_classification.category_code, "SAN-GARB")

        # Verify trace has 1 step accepted
        trace = res.secondary_signals[0].get("trace", [])
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["tool"], "TFIDF")
        self.assertEqual(trace[0]["decision"], "ACCEPT")

    def test_2_low_confidence_tfidf_escalates_to_secondary_tool(self):
        """Verify low-confidence TF-IDF escalates to embedding / fallback tool."""
        agent = ClassificationAgent(taxonomy_manager=self.tm, tfidf_threshold=0.99, tfidf_margin=0.50)
        text = "Garbage truck missing today."
        understanding = self.understanding_agent.understand(text)
        res = agent.classify(understanding, debug=True)

        trace = res.secondary_signals[0].get("trace", [])
        self.assertGreaterEqual(len(trace), 2)
        self.assertEqual(trace[0]["tool"], "TFIDF")
        self.assertEqual(trace[0]["decision"], "ESCALATE")

    def test_3_out_of_scope_query_handling(self):
        """Verify non-municipal out-of-scope query is correctly flagged OUT_OF_SCOPE."""
        text = "My mobile phone battery drains very fast and cricket match score."
        understanding = self.understanding_agent.understand(text)
        res = self.classification_agent.classify(understanding)

        self.assertEqual(res.status, ClassificationStatus.OUT_OF_SCOPE.value)
        self.assertEqual(res.administrative_mapping.type, MappingType.UNMAPPED.value)
        self.assertIsNone(res.semantic_classification.category_code)

    def test_4_insufficient_information_query_handling(self):
        """Verify vague/short query is correctly flagged INSUFFICIENT_INFORMATION."""
        text = "Fix"
        understanding = self.understanding_agent.understand(text)
        res = self.classification_agent.classify(understanding)

        self.assertEqual(res.status, ClassificationStatus.INSUFFICIENT_INFORMATION.value)
        self.assertEqual(res.administrative_mapping.type, MappingType.UNMAPPED.value)

    def test_5_llm_fallback_mock_integration(self):
        """Verify LLM fallback tool output adheres to taxonomy constraints."""
        llm_tool = LlmFallbackTool(taxonomy_manager=self.tm)
        res = llm_tool.classify(
            text="Complex indirect problem near bus stop",
            tfidf_candidates=[{"code": "SAN-GARB", "score": 0.40}],
            embedding_candidates=[{"code": "SAN-GARB", "score": 0.50}]
        )
        self.assertIn("status", res)
        self.assertIn("domain", res)
        self.assertIn("category", res)
        self.assertEqual(res["method"], "LLM_FALLBACK")

    def test_6_agent_trace_generation(self):
        """Verify ClassificationAgent generates structured, human-explainable trace log."""
        text = "Rastyavarchi street light maagchya aathavdya pasun chalat nahiye."
        understanding = self.understanding_agent.understand(text)
        res = self.classification_agent.classify(understanding, debug=True)

        self.assertIsNotNone(res.secondary_signals)
        self.assertTrue(len(res.secondary_signals) > 0)
        self.assertIn("trace", res.secondary_signals[0])

    def test_7_taxonomy_constrained_output(self):
        """Verify category and domain returned by agent exist in taxonomy.json."""
        text = "Water main pipeline leaking heavily near signal."
        understanding = self.understanding_agent.understand(text)
        res = self.classification_agent.classify(understanding)

        cat_code = res.semantic_classification.category_code
        dom_code = res.semantic_classification.domain_code

        self.assertIsNotNone(self.tm.get_category(cat_code))
        self.assertIsNotNone(self.tm.get_domain(dom_code))

    def test_8_administrative_mapping_fidelity(self):
        """Verify administrative mapping is correctly attached without conflating semantic domain."""
        text = "Commercial building emergency fire exit staircase blocked."
        understanding = self.understanding_agent.understand(text)
        res = self.classification_agent.classify(understanding)

        self.assertEqual(res.semantic_classification.domain_code, "FIRE_DISASTER_SAFETY")
        self.assertEqual(res.administrative_mapping.type, MappingType.CROSS_AGENCY.value)
        self.assertIsNone(res.administrative_mapping.department_code)


if __name__ == "__main__":
    unittest.main()
