"""
Phase 2 Unit Test Suite for CivicFlow AI TF-IDF Baseline.
Run via: py -m unittest discover -s tests -p "test_*.py"
"""

import csv
import sys
import unittest
from pathlib import Path

# Ensure root workspace directory in sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.taxonomy.taxonomy_manager import TaxonomyManager
from ai.models.tfidf_model import HierarchicalTfidfModel
from ai.tools.tfidf_tool import TfidfClassificationTool


class TestTfidfBaselinePhase2(unittest.TestCase):
    """Test cases for dataset loading, leakage prevention, hierarchical TF-IDF inference, and taxonomy integrity."""

    def setUp(self):
        self.processed_dir = Path(__file__).parent.parent / "ai" / "data" / "processed"
        self.tm = TaxonomyManager()
        self.tfidf_tool = TfidfClassificationTool()

    def test_1_dataset_processed_files_exist(self):
        """Verify train.csv, validation.csv, test.csv, and special_cases.csv exist."""
        for fname in ["train.csv", "validation.csv", "test.csv", "special_cases.csv"]:
            path = self.processed_dir / fname
            self.assertTrue(path.exists(), f"Processed dataset file missing: {fname}")

    def test_2_dataset_contains_all_taxonomy_categories(self):
        """Verify dataset covers all 34 taxonomy categories."""
        train_path = self.processed_dir / "train.csv"
        categories_in_train = set()

        with open(train_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                categories_in_train.add(row["category_code"])

        all_categories = set(self.tm.get_categories().keys())
        self.assertEqual(categories_in_train, all_categories, "Train dataset missing categories")

    def test_3_no_obvious_train_test_leakage(self):
        """Verify no duplicate complaint texts cross between train and test splits."""
        train_texts = set()
        test_texts = set()

        with open(self.processed_dir / "train.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                train_texts.add(row["complaint_text"].strip().lower())

        with open(self.processed_dir / "test.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                test_texts.add(row["complaint_text"].strip().lower())

        overlap = train_texts.intersection(test_texts)
        self.assertEqual(len(overlap), 0, f"Found {len(overlap)} leaking texts between train and test")

    def test_4_tfidf_model_loads_artifacts(self):
        """Verify TF-IDF model artifacts load successfully."""
        model = HierarchicalTfidfModel()
        loaded = model.load_artifacts()
        self.assertTrue(loaded, "Failed to load TF-IDF model artifacts")
        self.assertIsNotNone(model.domain_classifier)

    def test_5_domain_prediction_works(self):
        """Verify Level 1 Domain prediction on test sentence."""
        sample_text = "Garbage truck has not arrived for three days in society."
        pred = self.tfidf_tool.classify(sample_text)
        self.assertEqual(pred["domain"]["code"], "SOLID_WASTE")
        self.assertGreater(pred["domain"]["confidence"], 0.70)

    def test_6_category_prediction_works(self):
        """Verify Level 2 Category prediction on test sentence."""
        sample_text = "Huge dangerous pothole near signal damaging bikes."
        pred = self.tfidf_tool.classify(sample_text)
        self.assertEqual(pred["domain"]["code"], "ROADS_FOOTPATHS")
        self.assertEqual(pred["category"]["code"], "RDW-POT")
        self.assertGreater(pred["overall_confidence"], 0.70)

    def test_7_category_belongs_to_predicted_domain(self):
        """Verify predicted category strictly belongs to the predicted domain in taxonomy."""
        sample_text = "Main water pipeline burst near bus stop spilling water."
        pred = self.tfidf_tool.classify(sample_text)
        dom_code = pred["domain"]["code"]
        cat_code = pred["category"]["code"]

        domain_info = self.tm.get_domain(dom_code)
        self.assertIsNotNone(domain_info)
        domain_cats = domain_info.get("categories", {}).keys()
        self.assertIn(cat_code, domain_cats, f"Category {cat_code} does not belong to predicted domain {dom_code}")

    def test_8_inference_output_follows_contract(self):
        """Verify TfidfClassificationTool output matches classification output contract."""
        sample_text = "Rastyavarchi street light maagchya aathavdya pasun chalat nahiye."
        pred = self.tfidf_tool.classify(sample_text)

        self.assertIn("status", pred)
        self.assertIn("domain", pred)
        self.assertIn("category", pred)
        self.assertIn("administrative_mapping", pred)
        self.assertIn("overall_confidence", pred)
        self.assertIn("top_predictions", pred)
        self.assertEqual(pred["method"], "TFIDF")

        # Street lighting maps DIRECT to ELE
        self.assertEqual(pred["administrative_mapping"]["type"], "DIRECT")
        self.assertEqual(pred["administrative_mapping"]["department_code"], "ELE")

    def test_9_special_statuses_not_in_supervised_training(self):
        """Verify special cases (AMBIGUOUS, OUT_OF_SCOPE) are kept separate from train.csv."""
        special_path = self.processed_dir / "special_cases.csv"
        self.assertTrue(special_path.exists())

        special_diffs = set()
        with open(special_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                special_diffs.add(row["difficulty"])

        self.assertIn("AMBIGUOUS", special_diffs)
        self.assertIn("OUT_OF_SCOPE", special_diffs)
        self.assertIn("INSUFFICIENT_INFORMATION", special_diffs)

        # Confirm train.csv does not contain special difficulty cases
        with open(self.processed_dir / "train.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.assertNotIn(row["difficulty"], ["AMBIGUOUS", "OUT_OF_SCOPE", "INSUFFICIENT_INFORMATION"])

    def test_10_unmapped_and_cross_agency_in_inference(self):
        """Verify inference output correctly returns UNMAPPED/CROSS_AGENCY administrative mapping."""
        tax_text = "Property tax bill calculated twice the actual square footage in bill."
        pred_tax = self.tfidf_tool.classify(tax_text)
        self.assertEqual(pred_tax["domain"]["code"], "PROPERTY_TAX")
        self.assertEqual(pred_tax["administrative_mapping"]["type"], "UNMAPPED")
        self.assertIsNone(pred_tax["administrative_mapping"]["department_code"])

        fire_text = "Old dilapidated building balcony wall cracked severely and leaning over street."
        pred_fire = self.tfidf_tool.classify(fire_text)
        self.assertEqual(pred_fire["domain"]["code"], "FIRE_DISASTER_SAFETY")
        self.assertEqual(pred_fire["administrative_mapping"]["type"], "CROSS_AGENCY")
        self.assertIsNone(pred_fire["administrative_mapping"]["department_code"])


if __name__ == "__main__":
    unittest.main()
