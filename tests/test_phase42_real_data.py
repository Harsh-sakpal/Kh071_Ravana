"""
Phase 4.2 Unit Test Suite for Real Municipal Data Validation & Mapping.
Run via: py -m unittest discover -s tests -p "test_*.py"
"""

import csv
import sys
import unittest
from pathlib import Path

# Ensure root workspace in sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.taxonomy.taxonomy_manager import TaxonomyManager
from ai.data.create_source_mapping import SourceCategoryMapper
from ai.models.tfidf_model import HierarchicalTfidfModel


class TestPhase42RealMunicipalData(unittest.TestCase):
    """Test suite covering real municipal data mapping, split integrity, and model evaluation."""

    def setUp(self):
        self.tm = TaxonomyManager()
        self.mapper = SourceCategoryMapper()
        self.real_dir = Path(__file__).parent.parent / "ai" / "data" / "real"
        self.mapping_file = Path(__file__).parent.parent / "ai" / "data" / "mappings" / "source_to_civicflow_mapping.csv"

    def test_1_mapping_file_exists_and_valid(self):
        """Verify source_to_civicflow_mapping.csv exists and contains required schema."""
        self.assertTrue(self.mapping_file.exists())
        with open(self.mapping_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            self.assertIn("source_file", fieldnames)
            self.assertIn("civicflow_domain", fieldnames)
            self.assertIn("civicflow_category", fieldnames)
            self.assertIn("mapping_status", fieldnames)

    def test_2_unmapped_categories_rejected(self):
        """Verify administrative/non-civic categories are mapped as UNMAPPED."""
        dom, cat, conf, status, reason = self.mapper.get_mapping("CORONA COVID19", "Non availability of vaccines")
        self.assertEqual(status, "UNMAPPED")
        self.assertEqual(dom, "UNMAPPED")
        self.assertEqual(cat, "UNMAPPED")

    def test_3_known_category_mapping_validity(self):
        """Verify valid municipal complaint pairs map to exact taxonomy codes."""
        dom, cat, conf, status, reason = self.mapper.get_mapping("Electrical", "Street Light Not Working")
        self.assertEqual(status, "MAPPED")
        self.assertEqual(dom, "STREET_LIGHTING")
        self.assertEqual(cat, "ELE-LGT")

        dom2, cat2, conf2, status2, reason2 = self.mapper.get_mapping("Road Maintenance(Engg)", "Potholes")
        self.assertEqual(status2, "MAPPED")
        self.assertEqual(dom2, "ROADS_FOOTPATHS")
        self.assertEqual(cat2, "RDW-POT")

    def test_4_real_data_splits_exist_and_clean(self):
        """Verify real_train.csv and real_validation.csv exist and have non-zero rows."""
        train_path = self.real_dir / "real_train.csv"
        val_path = self.real_dir / "real_validation.csv"
        self.assertTrue(train_path.exists())
        self.assertTrue(val_path.exists())

        with open(train_path, "r", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        with open(val_path, "r", encoding="utf-8") as f:
            val_rows = list(csv.DictReader(f))

        self.assertGreater(len(train_rows), 1000)
        self.assertGreater(len(val_rows), 100)

    def test_5_real_split_leakage_prevention(self):
        """Verify zero text overlap between real_train.csv and real_validation.csv."""
        train_path = self.real_dir / "real_train.csv"
        val_path = self.real_dir / "real_validation.csv"

        train_texts = set()
        with open(train_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                train_texts.add(r["complaint_text"].strip().lower())

        overlap_count = 0
        with open(val_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["complaint_text"].strip().lower() in train_texts:
                    overlap_count += 1

        self.assertEqual(overlap_count, 0)

    def test_6_real_trained_model_artifacts(self):
        """Verify real-trained model loads successfully from ai/models/tfidf_real/."""
        real_models_dir = Path(__file__).parent.parent / "ai" / "models" / "tfidf_real"
        self.assertTrue(real_models_dir.exists())

        model = HierarchicalTfidfModel(artifacts_dir=real_models_dir)
        loaded = model.load_artifacts()
        self.assertTrue(loaded)

        res = model.predict("Potholes issue in Katraj area.")
        self.assertEqual(res["domain"]["code"], "ROADS_FOOTPATHS")
        self.assertEqual(res["category"]["code"], "RDW-POT")


if __name__ == "__main__":
    unittest.main()
