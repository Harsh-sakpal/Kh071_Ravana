"""
Unit test suite for CivicFlow AI AI/ML/NLP foundation (Person 1 Scope).
Run via: py -m unittest discover -s tests -p "test_*.py"
"""

import sys
import unittest
from pathlib import Path

# Add root workspace directory to sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.taxonomy.taxonomy_manager import TaxonomyManager
from ai.agents.understanding_agent import UnderstandingAgent, LanguageDetector
from ai.agents.classification_agent import ClassificationAgent
from ai.validation import CivicFlowDataValidator
from ai.schemas.dataclasses import (
    UnderstandingOutput,
    ClassificationOutput,
    ClassificationStatus,
    MappingType,
)


class TestCivicFlowAIFoundation(unittest.TestCase):
    """Test cases for AI taxonomy, Understanding Agent, schemas, and validation."""

    def setUp(self):
        self.taxonomy_manager = TaxonomyManager()
        self.understanding_agent = UnderstandingAgent()
        self.classification_agent = ClassificationAgent(self.taxonomy_manager)
        self.validator = CivicFlowDataValidator(self.taxonomy_manager)

    def test_1_taxonomy_loads_correctly(self):
        """Verify taxonomy loads without error and contains metadata."""
        self.assertIsNotNone(self.taxonomy_manager.data)
        self.assertIn("domains", self.taxonomy_manager.data)
        self.assertIn("departments", self.taxonomy_manager.data)

    def test_2_all_15_domains_exist(self):
        """Verify all 15 fixed AI domains exist in taxonomy."""
        domains = self.taxonomy_manager.get_domains()
        self.assertEqual(len(domains), 15)
        for fixed_domain in TaxonomyManager.FIXED_DOMAINS:
            self.assertIn(fixed_domain, domains)

    def test_3_preserved_category_codes_exist(self):
        """Verify all six preserved category codes exist in taxonomy with correct definitions."""
        preserved = [
            "SAN-GARB",
            "SAN-DUMP",
            "RDW-POT",
            "RDW-WALK",
            "ELE-LGT",
            "WTR-LEAK",
        ]
        categories = self.taxonomy_manager.get_categories()
        for code in preserved:
            self.assertIn(code, categories, f"Preserved code {code} missing from taxonomy")
            cat_info = categories[code]
            dept_map = cat_info.get("department_mapping", {})
            self.assertEqual(dept_map.get("type"), "DIRECT")
            self.assertIn(dept_map.get("department_code"), ["SAN", "RDW", "ELE", "WTR", "PRK"])

    def test_4_understanding_agent_accepts_english(self):
        """Verify Understanding Agent processes English input correctly."""
        text = "Garbage vehicle has not arrived for three days in society."
        result = self.understanding_agent.understand(text)
        self.assertIsInstance(result, UnderstandingOutput)
        self.assertEqual(result.original_text, text)
        self.assertEqual(result.language.code, "EN")
        self.assertGreaterEqual(result.language.confidence, 0.7)

    def test_5_understanding_agent_accepts_marathi(self):
        """Verify Understanding Agent processes Marathi input correctly."""
        text = "आमच्या गल्लीत कचरा कुंडी तुटली आहे आणि कचरा रस्त्यावर आला आहे."
        result = self.understanding_agent.understand(text)
        self.assertIsInstance(result, UnderstandingOutput)
        self.assertEqual(result.original_text, text)
        self.assertEqual(result.language.code, "MARATHI")
        self.assertGreaterEqual(result.language.confidence, 0.8)

    def test_6_understanding_agent_accepts_hinglish(self):
        """Verify Understanding Agent processes Hinglish input correctly."""
        text = "Rastyavarchi street light maagchya aathavdya pasun chalat nahiye."
        result = self.understanding_agent.understand(text)
        self.assertIsInstance(result, UnderstandingOutput)
        self.assertEqual(result.original_text, text)
        self.assertEqual(result.language.code, "HINGLISH")
        self.assertGreaterEqual(result.language.confidence, 0.7)

    def test_7_original_text_is_preserved(self):
        """Verify original text is strictly preserved verbatim."""
        raw_text = "   URGENT: Pothole at 5th Cross Road!  \n\t"
        result = self.understanding_agent.understand(raw_text)
        self.assertEqual(result.original_text, raw_text)

    def test_8_normalization_works_without_destroying_meaning(self):
        """Verify text normalization is clean and conservative."""
        raw_text = "Pothole   near   Railway Station (Ward 12)"
        normalized = self.understanding_agent.normalize_text(raw_text)
        self.assertEqual(normalized, "pothole near railway station (ward 12)")
        self.assertIn("railway station", normalized)

    def test_9_ambiguous_complaint_handling(self):
        """Verify ambiguous complaint text does not receive a fabricated category."""
        vague_text = "Fix"
        understanding = self.understanding_agent.understand(vague_text)
        self.assertIn("INSUFFICIENT_DETAILS: Complaint text is extremely short and vague", understanding.missing_information)

        classification = self.classification_agent.classify(understanding)
        self.assertEqual(classification.status, ClassificationStatus.INSUFFICIENT_INFORMATION.value)
        self.assertIsNone(classification.semantic_classification.category_code)

    def test_10_out_of_scope_or_empty_complaint_representation(self):
        """Verify empty or invalid text produces appropriate classification status representation."""
        empty_text = ""
        understanding = self.understanding_agent.understand(empty_text)
        classification = self.classification_agent.classify(understanding)
        self.assertEqual(classification.status, ClassificationStatus.INSUFFICIENT_INFORMATION.value)
        self.assertEqual(classification.administrative_mapping.type, MappingType.UNMAPPED.value)
        self.assertIsNone(classification.administrative_mapping.department_code)

    def test_11_unmapped_and_cross_agency_domains(self):
        """Verify PROPERTY_TAX, LICENSES_PERMISSIONS, and FIRE_DISASTER_SAFETY mappings."""
        tax_cat = self.taxonomy_manager.get_category("TAX-ERR")
        self.assertIsNotNone(tax_cat)
        self.assertEqual(tax_cat["department_mapping"]["type"], "UNMAPPED")
        self.assertIsNone(tax_cat["department_mapping"]["department_code"])

        lic_cat = self.taxonomy_manager.get_category("LIC-TRD")
        self.assertIsNotNone(lic_cat)
        self.assertEqual(lic_cat["department_mapping"]["type"], "UNMAPPED")
        self.assertIsNone(lic_cat["department_mapping"]["department_code"])

        fire_cat = self.taxonomy_manager.get_category("SAF-FIRE")
        self.assertIsNotNone(fire_cat)
        self.assertEqual(fire_cat["department_mapping"]["type"], "CROSS_AGENCY")
        self.assertIsNone(fire_cat["department_mapping"]["department_code"])

    def test_12_validation_utility_passes(self):
        """Verify CivicFlowDataValidator passes taxonomy and seed dataset."""
        result = self.validator.run_all_checks()
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
