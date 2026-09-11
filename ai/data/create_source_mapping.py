"""
Source Category to CivicFlow Taxonomy Mapping Engine (Phase 4.2).

Single Source of Truth: ai/taxonomy/taxonomy.json
Maps real municipal complaint Category/Sub-Category pairs to 15 AI Domains and 34 Categories.
Unmapped / Administrative categories are explicitly marked UNMAPPED.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure root workspace in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.taxonomy.taxonomy_manager import TaxonomyManager


class SourceCategoryMapper:
    """Maps BBMP / Real Municipal dataset categories to CivicFlow Taxonomy."""

    # Explicit Mapping Rules based on exact Sub Category & Category text
    MAPPING_RULES = {
        # Solid Waste Management
        ("Solid Waste (Garbage) Related", "Garbage dumping in vacant sites"): ("SOLID_WASTE", "SAN-DUMP", 0.95, "MAPPED", "Direct match for open dumping"),
        ("Solid Waste (Garbage) Related", "Garbage dump"): ("SOLID_WASTE", "SAN-DUMP", 0.95, "MAPPED", "Direct match for garbage dump"),
        ("Solid Waste (Garbage) Related", "Public dustbins not cleaned"): ("SOLID_WASTE", "SAN-BIN", 0.95, "MAPPED", "Direct match for overflow/uncleaned public dustbin"),
        ("Solid Waste (Garbage) Related", "Garbage vehicle not arrived"): ("SOLID_WASTE", "SAN-GARB", 0.95, "MAPPED", "Direct match for uncollected door-to-door garbage"),
        ("Solid Waste (Garbage) Related", "Sweeping not done"): ("SOLID_WASTE", "SAN-GARB", 0.90, "MAPPED", "Street sweeping and waste collection"),
        ("Solid Waste (Garbage) Related", "Segregation of waste not done"): ("SOLID_WASTE", "SAN-SEG", 0.95, "MAPPED", "Direct match for waste segregation"),
        ("Solid Waste (Garbage) Related", "Burning of waste"): ("SOLID_WASTE", "SAN-BURN", 0.95, "MAPPED", "Direct match for open waste burning"),
        ("Solid Waste (Garbage) Related", "Debris dumping"): ("ROADS_FOOTPATHS", "RDW-BLDG", 0.90, "MAPPED", "Construction debris dumping on road/footpath"),
        ("Solid Waste (Garbage) Related", "Commercial Waste Dumping"): ("SOLID_WASTE", "SAN-BULK", 0.90, "MAPPED", "Bulk commercial waste dumping"),

        # Street Lighting / Electrical
        ("Electrical", "Street Light Not Working"): ("STREET_LIGHTING", "ELE-LGT", 0.98, "MAPPED", "Direct match for streetlight outage"),
        ("Electrical", "Requirement For New Street Lights"): ("STREET_LIGHTING", "ELE-LGT", 0.90, "MAPPED", "Streetlight installation request"),
        ("Electrical", "Street Lights Switched On During Day Time"): ("STREET_LIGHTING", "ELE-LGT", 0.90, "MAPPED", "Streetlight operational timing defect"),
        ("Electrical", "Open Electrical Junction Box"): ("STREET_LIGHTING", "ELE-POLE", 0.95, "MAPPED", "Exposed wiring and junction box hazard"),
        ("Electrical", "Earthing Issue relate to Electric Poles/Panel Boards"): ("STREET_LIGHTING", "ELE-POLE", 0.90, "MAPPED", "Damaged pole earthing hazard"),
        ("Electrical", "Park Lights Not Working"): ("STREET_LIGHTING", "ELE-LGT", 0.90, "MAPPED", "Park lighting outage"),

        # Roads & Footpaths
        ("Road Maintenance(Engg)", "Potholes"): ("ROADS_FOOTPATHS", "RDW-POT", 0.98, "MAPPED", "Direct match for pothole repair"),
        ("Road Maintenance(Engg)", "Footpath maintenance"): ("ROADS_FOOTPATHS", "RDW-WALK", 0.95, "MAPPED", "Footpath paving and walkability maintenance"),
        ("Road Maintenance(Engg)", "footpath encroachment"): ("ENCROACHMENT", "RDW-ENC", 0.95, "MAPPED", "Encroachment on public footpath/walkway"),
        ("Road Maintenance(Engg)", "Road cutting"): ("ROADS_FOOTPATHS", "RDW-RD", 0.90, "MAPPED", "Road cutting / asphalt damage"),
        ("Road Maintenance(Engg)", "Debris Removal / Construction Material"): ("ROADS_FOOTPATHS", "RDW-BLDG", 0.90, "MAPPED", "Construction debris on road"),
        ("Road Maintenance(Engg)", "Road side drains"): ("SEWERAGE_DRAINAGE", "SEW-DRAIN", 0.95, "MAPPED", "Roadside storm drain/gutter blockage"),
        ("Road Maintenance(Engg)", "water stagnation"): ("STORM_WATER", "STM-LOG", 0.95, "MAPPED", "Waterlogging / rainwater inundation"),
        ("Road Maintenance(Engg)", "UnAuthorised Constructions"): ("BUILDING_CONSTRUCTION", "BLD-UNAUTH", 0.95, "MAPPED", "Unauthorized construction on road line"),
        ("Road Maintenance(Engg)", "Overhanging cables"): ("STREET_LIGHTING", "ELE-POLE", 0.85, "MAPPED", "Low hanging overhead cables"),

        # Sewerage & Drainage
        ("Sanitation", "No electricity in public toilet(s)"): ("PUBLIC_HEALTH", "SAN-HYG", 0.85, "MAPPED", "Public toilet sanitation and maintenance"),
        ("Sanitation", "Public Toilet Maintenance"): ("PUBLIC_HEALTH", "SAN-HYG", 0.90, "MAPPED", "Public toilet sanitation"),
        ("Lakes", "Sewage inflow to lake"): ("SEWERAGE_DRAINAGE", "SEW-OVER", 0.90, "MAPPED", "Sewage overflow flowing into waterbody"),
        ("Lakes", "Lake not maintained properly"): ("ENVIRONMENT_POLLUTION", "ENV-WATER", 0.85, "MAPPED", "Waterbody pollution / lake degradation"),

        # Gardens & Trees
        ("Forest", "Removal of dead/fallen trees"): ("GARDENS_TREES", "PRK-TREE", 0.95, "MAPPED", "Fallen/dead tree removal"),
        ("Forest", "obstructions Branches / Trees."): ("GARDENS_TREES", "PRK-TREE", 0.95, "MAPPED", "Hazardous tree branch trimming"),
        ("Forest", "Tree Trimming"): ("GARDENS_TREES", "PRK-TREE", 0.95, "MAPPED", "Tree branch pruning"),
        ("Parks and Play grounds", "Play Grounds"): ("GARDENS_TREES", "PRK-MAINT", 0.90, "MAPPED", "Public park and playground maintenance"),

        # Animal Control / Public Health
        ("Forest", "Issues related Snakes/Birds/Monkeys"): ("PUBLIC_HEALTH", "SAN-ANM", 0.85, "MAPPED", "Animal control / wild animal issue"),
        ("veterinary", "Animal Rescue - Not well - Requires attention"): ("SOLID_WASTE", "SAN-ANM", 0.85, "MAPPED", "Dead/injured stray animal removal"),
        ("veterinary", "shifting of the street dogs to other places"): ("PUBLIC_HEALTH", "SAN-ANM", 0.85, "MAPPED", "Stray dog control"),
        ("veterinary", "Stray cattle related complaints"): ("PUBLIC_HEALTH", "SAN-ANM", 0.85, "MAPPED", "Stray cattle hazard"),
        ("veterinary", "Stray dogs related complaints"): ("PUBLIC_HEALTH", "SAN-ANM", 0.85, "MAPPED", "Stray dog nuisance"),

        # Building & Construction
        ("Town Planning", "Construction not done as per approved plan (Building Bye Law Violations)"): ("BUILDING_CONSTRUCTION", "BLD-UNAUTH", 0.95, "MAPPED", "Illegal construction / building plan violation"),
        ("Estate", "Encroachment"): ("ENCROACHMENT", "RDW-ENC", 0.90, "MAPPED", "Municipal land encroachment"),

        # Encroachment & Advertisements
        ("Advertisement", "Flex and banners"): ("ENCROACHMENT", "RDW-ENC", 0.90, "MAPPED", "Illegal flex/banner encroachment"),
        ("Advertisement", "Unsafe Hoardings"): ("FIRE_DISASTER_SAFETY", "SAF-DIS", 0.85, "MAPPED", "Unsafe hoarding collapse hazard"),

        # Environment & Health
        ("Plastic", "Use of banned single use plastic"): ("ENVIRONMENT_POLLUTION", "ENV-AIR", 0.80, "MAPPED", "Environmental single-use plastic ban enforcement"),
        ("Health Dept", "Unhygienic  premises in hospital"): ("PUBLIC_HEALTH", "SAN-HYG", 0.90, "MAPPED", "Public health hygiene violation"),
    }

    def __init__(self):
        self.tm = TaxonomyManager()

    def get_mapping(self, source_category: str, source_subcategory: str) -> Tuple[str, str, float, str, str]:
        """
        Returns (civicflow_domain, civicflow_category, confidence, mapping_status, reason)
        for a given source Category and Sub Category.
        """
        key = (source_category.strip(), source_subcategory.strip())
        if key in self.MAPPING_RULES:
            dom, cat, conf, status, reason = self.MAPPING_RULES[key]
            # Verify with TaxonomyManager single source of truth
            if self.tm.get_domain(dom) and self.tm.get_category(cat):
                return dom, cat, conf, status, reason

        # Check soft fallback rules
        sub_lower = source_subcategory.lower()
        cat_lower = source_category.lower()

        if "pothole" in sub_lower:
            return "ROADS_FOOTPATHS", "RDW-POT", 0.90, "MAPPED", "Soft match: pothole keyword"
        if "garbage" in sub_lower or "dustbin" in sub_lower or "sweeping" in sub_lower:
            return "SOLID_WASTE", "SAN-GARB", 0.85, "MAPPED", "Soft match: garbage keyword"
        if "street light" in sub_lower:
            return "STREET_LIGHTING", "ELE-LGT", 0.90, "MAPPED", "Soft match: streetlight keyword"
        if "drain" in sub_lower or "sewer" in sub_lower:
            return "SEWERAGE_DRAINAGE", "SEW-DRAIN", 0.85, "MAPPED", "Soft match: drain keyword"
        if "tree" in sub_lower or "branch" in sub_lower:
            return "GARDENS_TREES", "PRK-TREE", 0.90, "MAPPED", "Soft match: tree keyword"

        return "UNMAPPED", "UNMAPPED", 0.0, "UNMAPPED", "Administrative / Non-municipal category outside 15 AI domains"

    def build_mapping_file(self, source_files: list) -> Path:
        """Processes source files and generates ai/data/mappings/source_to_civicflow_mapping.csv."""
        mapping_dir = Path(__file__).parent / "mappings"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        out_csv = mapping_dir / "source_to_civicflow_mapping.csv"

        pairs_seen = set()
        rows_to_write = []

        for fp in source_files:
            if not os.path.exists(fp):
                continue
            file_name = os.path.basename(fp)
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    cat = r.get("Category", "").strip()
                    sub = r.get("Sub Category", "").strip()
                    pair = (file_name, cat, sub)
                    if pair not in pairs_seen:
                        pairs_seen.add(pair)
                        dom, civic_cat, conf, status, reason = self.get_mapping(cat, sub)
                        rows_to_write.append({
                            "source_file": file_name,
                            "source_category": cat,
                            "source_subcategory": sub,
                            "civicflow_domain": dom if status == "MAPPED" else "",
                            "civicflow_category": civic_cat if status == "MAPPED" else "",
                            "mapping_confidence": conf,
                            "mapping_status": status,
                            "reason": reason,
                        })

        fieldnames = ["source_file", "source_category", "source_subcategory", "civicflow_domain", "civicflow_category", "mapping_confidence", "mapping_status", "reason"]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_write)

        print(f"[Source Category Mapper] Generated mapping report ({len(rows_to_write)} rows) at {out_csv}")
        return out_csv


if __name__ == "__main__":
    source_files = [
        r"c:\Users\Harsh Sakpal\Downloads\413fa9ec-8d06-4ecb-884e-1436c5a0f5dd.csv",
        r"c:\Users\Harsh Sakpal\Downloads\82f88d50-71c5-4203-92ac-5ccb5cabc7a2.csv",
        r"c:\Users\Harsh Sakpal\Downloads\9e7e6892-06b6-4fdc-967a-e4787562f155.csv",
        r"c:\Users\Harsh Sakpal\Downloads\b0d6e9ff-5eef-48bf-ba86-985dbe8112d1.csv",
        r"c:\Users\Harsh Sakpal\Downloads\b4dd8dd1-1628-4f35-9247-ef5afaad214d.csv",
        r"c:\Users\Harsh Sakpal\Downloads\d4419a76-e2af-44b3-aa25-369c85126f0f.csv",
    ]
    mapper = SourceCategoryMapper()
    mapper.build_mapping_file(source_files)
