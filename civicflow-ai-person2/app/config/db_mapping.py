"""
CivicFlow AI - Domain -> Database Mapping
==========================================

Bridges the AI taxonomy (app.config.taxonomy) to the operational database
layer (departments, categories) described in the current migration package.

IMPORTANT (per Master Prompt Section 8):
    The 15 AI domains and 225 AI subcategories do NOT require the database
    to contain 225 categories. Where no `category_code` mapping exists yet,
    the Routing Agent must return the AI classification with
    `category_code: null` and flag the mapping as unavailable -- it must
    NEVER invent a database category.

Two layers of mapping are kept deliberately separate:

1. DOMAIN_TO_DEPARTMENT
   Every one of the 15 domains maps to exactly one of the 5 known
   departments. This is always resolvable, even if the fine-grained
   category is not yet seeded.

2. SUBCATEGORY_TO_CATEGORY_CODE
   Maps (domain, subcategory) -> an actual `categories.code` value that is
   confirmed present in the supplied migrations. Only rows that were
   unambiguously readable in the migration PDF are included here. Many
   INSERT statements in V16 were truncated in the source document, so this
   table is intentionally NOT exhaustive -- extend it only when a new
   category_code is confirmed against the real database, never guessed.
"""

from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Domain -> Department (always resolvable)
# ---------------------------------------------------------------------------
# Confirmed department codes from migration/V15__seed_development_data.sql:
#   SAN = Sanitation & Waste Management
#   RDW = Roads & Public Works
#   ELE = Electricity & Street Lighting
#   WTR = Water Supply & Sewage
#   PRK = Parks & Environment

DOMAIN_TO_DEPARTMENT: Dict[str, str] = {
    # Person 2 canonical domain keys
    "SOLID_WASTE": "SAN",
    "PUBLIC_HEALTH_SANITATION": "SAN",
    "PROPERTY_MUNICIPAL_TAX": "SAN",
    "LICENSING_PERMISSIONS": "SAN",
    "WATER_SUPPLY": "WTR",
    "SEWERAGE_DRAINAGE": "WTR",
    "STORM_WATER": "WTR",
    "ROADS_FOOTPATHS": "RDW",
    "BUILDING_CONSTRUCTION": "RDW",
    "ENCROACHMENT": "RDW",
    "TRAFFIC_TRANSPORT_PARKING": "RDW",
    "FIRE_DISASTER_EMERGENCY": "RDW",
    "STREET_LIGHTING_ELECTRICAL": "ELE",
    "GARDENS_GREENERY": "PRK",
    "ENVIRONMENT_POLLUTION": "PRK",

    # Person 1 taxonomy aliases (canonicalization bridge)
    "STREET_LIGHTING": "ELE",
    "GARDENS_TREES": "PRK",
    "PUBLIC_HEALTH": "SAN",
    "TRAFFIC_PARKING": "RDW",
    "PROPERTY_TAX": "SAN",
    "LICENSES_PERMISSIONS": "SAN",
    "FIRE_DISASTER_SAFETY": "RDW",
}
# NOTE: this domain->department assignment mirrors the grouping implied by
# ai_domains in V16 (e.g. DOM-SWM/DOM-PHS/DOM-PTX/DOM-LIC all under SAN;
# DOM-WS/DOM-SD/DOM-STW under WTR; DOM-RDF/DOM-BLD/DOM-ENC/DOM-TRF/DOM-FDS
# under RDW; DOM-SLE under ELE; DOM-GTR/DOM-ENV under PRK). Confirm against
# the live `ai_domains` table before treating as final -- several V16
# INSERT rows were truncated in the supplied document.

# ---------------------------------------------------------------------------
# 2. (Domain, Subcategory) -> category_code (only confirmed rows)
# ---------------------------------------------------------------------------
SUBCATEGORY_TO_CATEGORY_CODE: Dict[Tuple[str, str], str] = {
    # SOLID_WASTE (SAN)
    ("SOLID_WASTE", "GARBAGE_NOT_COLLECTED"): "SWM-001",
    ("SOLID_WASTE", "MISSED_GARBAGE_COLLECTION"): "SAN-GARB",
    ("SOLID_WASTE", "OVERFLOWING_GARBAGE_BIN"): "SAN-GARB",
    ("SOLID_WASTE", "ILLEGAL_DUMPING"): "SAN-DUMP",
    ("SOLID_WASTE", "DEAD_ANIMAL_WASTE"): "SWM-002",
    ("SOLID_WASTE", "LITTERING"): "SWM-003",
    # PUBLIC_HEALTH_SANITATION (SAN)
    ("PUBLIC_HEALTH_SANITATION", "MOSQUITO_BREEDING"): "PHS-001",
    ("PUBLIC_HEALTH_SANITATION", "PUBLIC_TOILET_ISSUE"): "PHS-002",
    ("PUBLIC_HEALTH_SANITATION", "PUBLIC_TOILET_CLEANING"): "PHS-002",
    ("PUBLIC_HEALTH_SANITATION", "STRAY_ANIMAL_ISSUE"): "PHS-003",
    # PROPERTY_MUNICIPAL_TAX (SAN)
    ("PROPERTY_MUNICIPAL_TAX", "PROPERTY_TAX_ASSESSMENT"): "PTX-001",
    ("PROPERTY_MUNICIPAL_TAX", "PROPERTY_TAX_PAYMENT"): "PTX-002",
    # LICENSING_PERMISSIONS (SAN)
    ("LICENSING_PERMISSIONS", "HOARDING_PERMISSION"): "LIC-001",
    ("LICENSING_PERMISSIONS", "SIGNAGE_PERMISSION"): "LIC-001",
    ("LICENSING_PERMISSIONS", "TRADE_LICENSE"): "LIC-002",
    ("LICENSING_PERMISSIONS", "LICENSE_RENEWAL"): "LIC-002",
    # WATER_SUPPLY (WTR)
    ("WATER_SUPPLY", "NO_WATER_SUPPLY"): "WS-001",
    ("WATER_SUPPLY", "WATER_CONTAMINATION"): "WS-002",
    ("WATER_SUPPLY", "DIRTY_WATER"): "WS-002",
    ("WATER_SUPPLY", "WATER_MAIN_LEAK"): "WTR-LEAK",
    ("WATER_SUPPLY", "PIPELINE_LEAK"): "WS-003",
    # SEWERAGE_DRAINAGE (WTR)
    ("SEWERAGE_DRAINAGE", "SEWAGE_OVERFLOW"): "SD-001",
    ("SEWERAGE_DRAINAGE", "OPEN_MANHOLE"): "SD-002",
    ("SEWERAGE_DRAINAGE", "MANHOLE_DAMAGE"): "SD-002",
    # STORM_WATER (WTR)
    ("STORM_WATER", "WATERLOGGING"): "STW-001",
    ("STORM_WATER", "FLOODED_ROAD"): "STW-001",
    ("STORM_WATER", "BLOCKED_STORM_DRAIN"): "STW-002",
    # ROADS_FOOTPATHS (RDW)
    ("ROADS_FOOTPATHS", "POTHOLE"): "RDW-POT",
    ("ROADS_FOOTPATHS", "DAMAGED_ROAD"): "RDF-001",
    ("ROADS_FOOTPATHS", "BROKEN_FOOTPATH"): "RDW-WALK",
    ("ROADS_FOOTPATHS", "FOOTPATH_DAMAGE"): "RDF-002",
    ("ROADS_FOOTPATHS", "SPEED_BREAKER_ISSUE"): "RDF-003",
    # BUILDING_CONSTRUCTION (RDW)
    ("BUILDING_CONSTRUCTION", "UNAUTHORIZED_CONSTRUCTION"): "BLD-001",
    ("BUILDING_CONSTRUCTION", "BUILDING_VIOLATION"): "BLD-001",
    ("BUILDING_CONSTRUCTION", "DANGEROUS_BUILDING"): "BLD-002",
    ("BUILDING_CONSTRUCTION", "STRUCTURAL_DAMAGE"): "BLD-002",
    # ENCROACHMENT (RDW)
    ("ENCROACHMENT", "FOOTPATH_ENCROACHMENT"): "ENC-001",
    ("ENCROACHMENT", "ROAD_ENCROACHMENT"): "ENC-001",
    # TRAFFIC_TRANSPORT_PARKING (RDW)
    ("TRAFFIC_TRANSPORT_PARKING", "ILLEGAL_PARKING"): "TRF-001",
    ("TRAFFIC_TRANSPORT_PARKING", "TRAFFIC_SIGNAL_ISSUE"): "TRF-002",
    # FIRE_DISASTER_EMERGENCY (RDW)
    ("FIRE_DISASTER_EMERGENCY", "EXPOSED_ELECTRICAL_WIRE"): "FDS-001",
    ("FIRE_DISASTER_EMERGENCY", "ELECTRICAL_HAZARD"): "FDS-001",
    ("FIRE_DISASTER_EMERGENCY", "FIRE_HAZARD"): "FDS-002",
    ("FIRE_DISASTER_EMERGENCY", "FIRE_SAFETY_VIOLATION"): "FDS-002",
    # STREET_LIGHTING_ELECTRICAL (ELE)
    ("STREET_LIGHTING_ELECTRICAL", "STREETLIGHT_OUTAGE"): "ELE-LGT",
    ("STREET_LIGHTING_ELECTRICAL", "DARK_STREET"): "SLE-001",
    ("STREET_LIGHTING_ELECTRICAL", "BROKEN_LIGHT_POLE"): "SLE-002",
    ("STREET_LIGHTING_ELECTRICAL", "DAMAGED_STREETLIGHT"): "SLE-002",
    # GARDENS_GREENERY (PRK)
    ("GARDENS_GREENERY", "FALLEN_TREE"): "GTR-001",
    ("GARDENS_GREENERY", "TREE_BRANCH_HAZARD"): "GTR-002",
    ("GARDENS_GREENERY", "PARK_MAINTENANCE"): "GTR-003",
    # ENVIRONMENT_POLLUTION (PRK)
    ("ENVIRONMENT_POLLUTION", "NOISE_POLLUTION"): "ENV-001",
    ("ENVIRONMENT_POLLUTION", "EXCESSIVE_NOISE"): "ENV-001",
    ("ENVIRONMENT_POLLUTION", "HAZARDOUS_WASTE"): "ENV-002",
    ("ENVIRONMENT_POLLUTION", "CHEMICAL_CONTAMINATION"): "ENV-002",
}


def department_for_domain(domain: str) -> Optional[str]:
    """Always resolvable for any valid domain in the locked taxonomy."""
    return DOMAIN_TO_DEPARTMENT.get(domain)


def category_code_for(domain: str, subcategory: str) -> Optional[str]:
    """Returns None (never a fabricated value) if no confirmed DB mapping exists."""
    return SUBCATEGORY_TO_CATEGORY_CODE.get((domain, subcategory))
