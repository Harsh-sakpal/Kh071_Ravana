"""
Validation Utility for CivicFlow AI Taxonomy & Datasets.

Executes validation checks verifying:
1. All 15 fixed AI domains exist in taxonomy.
2. Every category belongs to a valid domain.
3. Every category has a valid department mapping (DIRECT, UNMAPPED, CROSS_AGENCY).
4. All preserved category codes (SAN-GARB, SAN-DUMP, RDW-POT, RDW-WALK, ELE-LGT, WTR-LEAK) exist.
5. No duplicate category codes across domains.
6. Dataset labels in raw/seed datasets match valid taxonomy codes.
7. No invalid department codes are introduced.
"""

import json
import sys
from pathlib import Path

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Dict, Any, List
from ai.taxonomy.taxonomy_manager import TaxonomyManager


class CivicFlowDataValidator:
    """Validator for AI taxonomy integrity and complaint dataset label correctness."""

    def __init__(self, taxonomy_manager: TaxonomyManager = None):
        self.tm = taxonomy_manager or TaxonomyManager()

    def validate_taxonomy(self) -> Dict[str, Any]:
        """Runs taxonomy structure validation using TaxonomyManager."""
        return self.tm.validate_taxonomy()

    def validate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validates dataset file against taxonomy.
        Verifies domain_code, category_code, and department_code.
        """
        path = Path(dataset_path)
        if not path.exists():
            return {
                "valid": False,
                "errors": [f"Dataset file not found: {dataset_path}"],
                "total_records": 0,
            }

        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        errors = []
        warnings = []
        valid_domains = set(self.tm.get_domains())
        all_categories = self.tm.get_categories()

        for idx, rec in enumerate(records):
            rec_id = rec.get("id", f"INDEX-{idx}")
            domain = rec.get("domain_code")
            category = rec.get("category_code")
            dept = rec.get("department_code")
            diff = rec.get("difficulty")

            # Out of scope / Ambiguous may have null domain/category
            if diff in ("AMBIGUOUS", "OUT_OF_SCOPE") and domain is None and category is None:
                continue

            if domain and domain not in valid_domains:
                errors.append(f"Record {rec_id}: Invalid domain_code '{domain}'")

            if category:
                if category not in all_categories:
                    errors.append(f"Record {rec_id}: Invalid category_code '{category}'")
                else:
                    cat_info = all_categories[category]
                    expected_map = cat_info.get("department_mapping", {})
                    map_type = expected_map.get("type")
                    exp_dept = expected_map.get("department_code")

                    if map_type == "DIRECT":
                        if dept != exp_dept:
                            errors.append(
                                f"Record {rec_id}: Category '{category}' expects DIRECT department '{exp_dept}' but found '{dept}'"
                            )
                    elif map_type in ("UNMAPPED", "CROSS_AGENCY"):
                        if dept is not None:
                            errors.append(
                                f"Record {rec_id}: Category '{category}' has mapping '{map_type}' and expects null department, but found '{dept}'"
                            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_records": len(records),
        }

    def run_all_checks(self, seed_dataset_path: str = None) -> bool:
        """Runs taxonomy and dataset validation and prints report."""
        if seed_dataset_path is None:
            seed_dataset_path = str(Path(__file__).parent / "data" / "raw" / "seed_complaints.json")

        print("==================================================")
        print("CIVICFLOW AI - DATA & TAXONOMY VALIDATION")
        print("==================================================")

        tax_res = self.validate_taxonomy()
        print(f"Taxonomy Validation Result: {'PASSED' if tax_res['valid'] else 'FAILED'}")
        print(f"  - Total AI Domains: {tax_res['total_domains']} / 15")
        print(f"  - Total Categories: {tax_res['total_categories']}")

        if tax_res["errors"]:
            print("  Taxonomy Errors:")
            for err in tax_res["errors"]:
                print(f"    [X] {err}")

        ds_res = self.validate_dataset(seed_dataset_path)
        print(f"\nSeed Dataset Validation Result: {'PASSED' if ds_res['valid'] else 'FAILED'}")
        print(f"  - Total Dataset Records: {ds_res['total_records']}")

        if ds_res["errors"]:
            print("  Dataset Errors:")
            for err in ds_res["errors"]:
                print(f"    [X] {err}")

        all_passed = tax_res["valid"] and ds_res["valid"]
        print("\nOVERALL VALIDATION STATUS:", "PASS" if all_passed else "FAIL")
        print("==================================================")
        return all_passed


if __name__ == "__main__":
    validator = CivicFlowDataValidator()
    success = validator.run_all_checks()
    if not success:
        sys.exit(1)
