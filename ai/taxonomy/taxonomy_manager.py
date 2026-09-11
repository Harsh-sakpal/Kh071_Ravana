"""
Taxonomy Manager for CivicFlow AI.
Provides programmatic access to the 15-domain fixed AI taxonomy and department mappings.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class TaxonomyManager:
    """Manager for loading, querying, and validating the AI taxonomy."""

    PRESERVED_CATEGORY_CODES = {
        "SAN-GARB",
        "SAN-DUMP",
        "RDW-POT",
        "RDW-WALK",
        "ELE-LGT",
        "WTR-LEAK",
    }

    PRESERVED_DEPARTMENT_CODES = {
        "SAN",
        "RDW",
        "ELE",
        "WTR",
        "PRK",
    }

    FIXED_DOMAINS = [
        "SOLID_WASTE",
        "WATER_SUPPLY",
        "SEWERAGE_DRAINAGE",
        "STORM_WATER",
        "ROADS_FOOTPATHS",
        "STREET_LIGHTING",
        "GARDENS_TREES",
        "PUBLIC_HEALTH",
        "BUILDING_CONSTRUCTION",
        "ENCROACHMENT",
        "TRAFFIC_PARKING",
        "PROPERTY_TAX",
        "LICENSES_PERMISSIONS",
        "ENVIRONMENT_POLLUTION",
        "FIRE_DISASTER_SAFETY",
    ]

    def __init__(self, taxonomy_path: Optional[str] = None):
        if taxonomy_path is None:
            taxonomy_path = str(Path(__file__).parent / "taxonomy.json")
        self.taxonomy_path = Path(taxonomy_path)
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Loads taxonomy data from JSON file."""
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found at {self.taxonomy_path}")
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def get_domains(self) -> List[str]:
        """Returns list of domain codes."""
        return list(self.data.get("domains", {}).keys())

    def get_domain(self, domain_code: str) -> Optional[Dict[str, Any]]:
        """Returns domain details by code."""
        return self.data.get("domains", {}).get(domain_code)

    def get_categories(self, domain_code: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Returns dictionary of all categories, optionally filtered by domain_code.
        Map key is category_code.
        """
        categories = {}
        domains = self.data.get("domains", {})
        if domain_code:
            domain_info = domains.get(domain_code, {})
            return domain_info.get("categories", {})
        
        for d_code, d_info in domains.items():
            for c_code, c_info in d_info.get("categories", {}).items():
                categories[c_code] = c_info
        return categories

    def get_category(self, category_code: str) -> Optional[Dict[str, Any]]:
        """Finds category information by category code across all domains."""
        all_cats = self.get_categories()
        return all_cats.get(category_code)

    def get_department_mapping(self, category_code: str) -> Optional[Dict[str, Any]]:
        """Retrieves administrative department mapping for a category code."""
        cat = self.get_category(category_code)
        if cat:
            return cat.get("department_mapping")
        return None

    def validate_taxonomy(self) -> Dict[str, Any]:
        """
        Validates structure integrity of the loaded taxonomy.
        Returns validation report dict.
        """
        errors = []
        warnings = []
        
        domains = self.data.get("domains", {})
        
        # Check all 15 domains exist
        for fixed_domain in self.FIXED_DOMAINS:
            if fixed_domain not in domains:
                errors.append(f"Missing fixed AI domain: {fixed_domain}")

        all_category_codes = set()
        for d_code, d_info in domains.items():
            for c_code, c_info in d_info.get("categories", {}).items():
                if c_code in all_category_codes:
                    errors.append(f"Duplicate category code found: {c_code}")
                all_category_codes.add(c_code)

                # Check department mapping
                dept_map = c_info.get("department_mapping", {})
                map_type = dept_map.get("type")
                dept_code = dept_map.get("department_code")

                if map_type == "DIRECT":
                    if dept_code not in self.PRESERVED_DEPARTMENT_CODES:
                        errors.append(f"Category {c_code} has invalid DIRECT department code: {dept_code}")
                elif map_type in ("UNMAPPED", "CROSS_AGENCY"):
                    if dept_code is not None:
                        errors.append(f"Category {c_code} has map_type {map_type} but department_code is not null")
                else:
                    errors.append(f"Category {c_code} has unknown mapping type: {map_type}")

        # Check preserved categories exist
        for pres_code in self.PRESERVED_CATEGORY_CODES:
            if pres_code not in all_category_codes:
                errors.append(f"Preserved category code missing: {pres_code}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_domains": len(domains),
            "total_categories": len(all_category_codes),
        }
