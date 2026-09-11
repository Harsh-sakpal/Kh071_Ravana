"""
Dataset Preparation & Stratified Splitter for CivicFlow AI.

Executes dataset generation, data validation, leak-prevention grouping,
and stratified train (70%) / validation (15%) / test (15%) splitting.
Outputs processed CSV files to ai/data/processed/.
"""

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure workspace root in sys.path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.data.generator import MunicipalDatasetGenerator
from ai.validation import CivicFlowDataValidator


class DatasetPreparer:
    """Handles validation, leak prevention grouping, and stratified splitting."""

    def __init__(self, raw_dir: Path = None, processed_dir: Path = None):
        self.raw_dir = raw_dir or (Path(__file__).parent / "raw")
        self.processed_dir = processed_dir or (Path(__file__).parent / "processed")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def prepare_and_split(self, seed: int = 42) -> Tuple[int, int, int]:
        """Main execution flow for dataset generation, validation, and splitting."""
        random.seed(seed)
        
        gen = MunicipalDatasetGenerator()
        synthetic_path = self.raw_dir / "synthetic_complaints.json"
        special_path = self.raw_dir / "special_cases.json"

        # Generate data
        print("[1/5] Generating synthetic municipal complaint dataset...")
        cat_records = gen.generate_category_dataset(target_per_category=100, seed=seed)
        special_records = gen.generate_special_cases_dataset(target_per_status=50, seed=seed)

        with open(synthetic_path, "w", encoding="utf-8") as f:
            json.dump(cat_records, f, indent=2, ensure_ascii=False)

        with open(special_path, "w", encoding="utf-8") as f:
            json.dump(special_records, f, indent=2, ensure_ascii=False)

        # Validate dataset
        print("[2/5] Validating dataset integrity with CivicFlowDataValidator...")
        validator = CivicFlowDataValidator()
        val_res = validator.validate_dataset(str(synthetic_path))

        if not val_res["valid"]:
            print("Dataset validation failed with errors:")
            for err in val_res["errors"]:
                print(f"  [X] {err}")
            raise ValueError("Dataset validation failed.")
        print(f"  [OK] Synthetic dataset validated: {val_res['total_records']} records.")

        # Group by category, then by unique complaint_text to prevent train/test leakage
        print("[3/5] Performing leak-prevention text-grouping and stratified 70/15/15 split...")
        category_text_groups = defaultdict(lambda: defaultdict(list))
        for rec in cat_records:
            cat_code = rec["category_code"]
            txt_key = rec["complaint_text"].strip().lower()
            category_text_groups[cat_code][txt_key].append(rec)

        train_records = []
        val_records = []
        test_records = []

        for cat_code, text_dict in category_text_groups.items():
            # Get list of unique text groups for this category
            unique_text_groups = list(text_dict.values())
            random.shuffle(unique_text_groups)

            n_total_groups = len(unique_text_groups)
            n_train_groups = int(n_total_groups * 0.70)
            n_val_groups = int(n_total_groups * 0.15)

            train_groups = unique_text_groups[:n_train_groups]
            val_groups = unique_text_groups[n_train_groups:n_train_groups + n_val_groups]
            test_groups = unique_text_groups[n_train_groups + n_val_groups:]

            for group in train_groups:
                train_records.extend(group)
            for group in val_groups:
                val_records.extend(group)
            for group in test_groups:
                test_records.extend(group)

        # Shuffle each split
        random.shuffle(train_records)
        random.shuffle(val_records)
        random.shuffle(test_records)

        # Save CSVs
        print("[4/5] Writing processed CSV datasets to ai/data/processed/...")
        fields = ["id", "complaint_text", "domain_code", "category_code", "department_code", "language", "difficulty", "source"]

        def save_csv(filename: str, records: List[Dict[str, Any]]):
            path = self.processed_dir / filename
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in records:
                    row = {k: r.get(k) for k in fields}
                    writer.writerow(row)

        save_csv("train.csv", train_records)
        save_csv("validation.csv", val_records)
        save_csv("test.csv", test_records)
        save_csv("special_cases.csv", special_records)

        print("[5/5] Dataset preparation successfully completed!")
        print(f"  - Train records:      {len(train_records)} ({len(train_records)/len(cat_records):.1%})")
        print(f"  - Validation records: {len(val_records)} ({len(val_records)/len(cat_records):.1%})")
        print(f"  - Test records:       {len(test_records)} ({len(test_records)/len(cat_records):.1%})")
        print(f"  - Special cases:      {len(special_records)}")

        # Print language & category breakdown
        langs = Counter(r["language"] for r in cat_records)
        print("  - Language Breakdown:")
        for lang, count in langs.items():
            print(f"      {lang}: {count} ({count/len(cat_records):.1%})")

        return len(train_records), len(val_records), len(test_records)


if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.prepare_and_split()
