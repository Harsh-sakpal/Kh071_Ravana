"""
CivicFlow AI - Canonical Civic Domain Taxonomy
================================================

LOCKED. Do not silently modify. This taxonomy is authoritative per the
Person-2 Master Development Prompt, Section 7.

Structure:
    DOMAIN (15 top-level)
        -> SUBCATEGORY (15 each, 225 total)

This module owns ONLY the taxonomy itself. Database mappings (domain ->
category_code -> department_code) live in `db_mapping.py` so the two
concerns never get tangled.
"""

from typing import Dict, List

TAXONOMY: Dict[str, List[str]] = {
    "SOLID_WASTE": [
        "GARBAGE_NOT_COLLECTED",
        "OVERFLOWING_GARBAGE_BIN",
        "ILLEGAL_DUMPING",
        "WASTE_SEGREGATION",
        "MISSED_GARBAGE_COLLECTION",
        "GARBAGE_VEHICLE_ISSUE",
        "PUBLIC_WASTE_BIN_ISSUE",
        "BULK_WASTE_DISPOSAL",
        "CONSTRUCTION_WASTE",
        "DEAD_ANIMAL_WASTE",
        "LITTERING",
        "WASTE_COLLECTION_SCHEDULE",
        "DOOR_TO_DOOR_COLLECTION",
        "WASTE_PROCESSING_ISSUE",
        "OTHER_SOLID_WASTE",
    ],
    "WATER_SUPPLY": [
        "NO_WATER_SUPPLY",
        "LOW_WATER_PRESSURE",
        "DIRTY_WATER",
        "WATER_MAIN_LEAK",
        "PIPELINE_LEAK",
        "PIPELINE_DAMAGE",
        "WATER_CONTAMINATION",
        "IRREGULAR_WATER_SUPPLY",
        "WATER_TANKER_REQUEST",
        "VALVE_ISSUE",
        "WATER_METER_ISSUE",
        "ILLEGAL_WATER_CONNECTION",
        "PIPELINE_BLOCKAGE",
        "WATER_SUPPLY_SCHEDULE",
        "OTHER_WATER_SUPPLY",
    ],
    "SEWERAGE_DRAINAGE": [
        "SEWAGE_OVERFLOW",
        "BLOCKED_SEWER",
        "SEWER_LEAKAGE",
        "MANHOLE_DAMAGE",
        "OPEN_MANHOLE",
        "MANHOLE_OVERFLOW",
        "DRAIN_BLOCKAGE",
        "DRAIN_DAMAGE",
        "FOUL_SEWAGE_ODOR",
        "SEWER_PIPE_DAMAGE",
        "ILLEGAL_SEWER_CONNECTION",
        "BACKFLOW",
        "SEWAGE_ACCUMULATION",
        "DRAIN_CLEANING_REQUEST",
        "OTHER_SEWERAGE_DRAINAGE",
    ],
    "STORM_WATER": [
        "WATERLOGGING",
        "FLOODED_ROAD",
        "BLOCKED_STORM_DRAIN",
        "STORM_DRAIN_DAMAGE",
        "DRAIN_OVERFLOW",
        "FLOODING_NEAR_PROPERTY",
        "RAINWATER_ACCUMULATION",
        "CULVERT_BLOCKAGE",
        "STORM_PIPE_BLOCKAGE",
        "STORM_DRAIN_CLEANING",
        "FLOOD_RISK_LOCATION",
        "POOR_STORM_WATER_FLOW",
        "DAMAGED_DRAIN_COVER",
        "MONSOON_WATER_ISSUE",
        "OTHER_STORM_WATER",
    ],
    "ROADS_FOOTPATHS": [
        "POTHOLE",
        "DAMAGED_ROAD",
        "BROKEN_FOOTPATH",
        "FOOTPATH_DAMAGE",
        "ROAD_SURFACE_DAMAGE",
        "ROAD_CRACK",
        "ROAD_CAVE_IN",
        "ROAD_SHOULDER_DAMAGE",
        "MISSING_FOOTPATH",
        "DAMAGED_ROAD_DIVIDER",
        "ROAD_MARKING_ISSUE",
        "SPEED_BREAKER_ISSUE",
        "OPEN_ROAD_EXCAVATION",
        "ROAD_REPAIR_DELAY",
        "OTHER_ROADS_FOOTPATHS",
    ],
    "STREET_LIGHTING_ELECTRICAL": [
        "STREETLIGHT_OUTAGE",
        "STREETLIGHT_FLICKERING",
        "DAMAGED_STREETLIGHT",
        "BROKEN_LIGHT_POLE",
        "EXPOSED_ELECTRICAL_WIRE",
        "ELECTRICAL_HAZARD",
        "SPARKING_EQUIPMENT",
        "DAMAGED_ELECTRICAL_BOX",
        "DARK_STREET",
        "STREETLIGHT_TIMER_ISSUE",
        "MULTIPLE_LIGHT_FAILURE",
        "ELECTRICAL_POLE_DAMAGE",
        "PUBLIC_ELECTRICAL_HAZARD",
        "STREETLIGHT_MAINTENANCE",
        "OTHER_STREET_LIGHTING_ELECTRICAL",
    ],
    "GARDENS_GREENERY": [
        "PARK_MAINTENANCE",
        "FALLEN_TREE",
        "TREE_PRUNING",
        "TREE_REMOVAL",
        "ILLEGAL_TREE_CUTTING",
        "DAMAGED_PARK_EQUIPMENT",
        "PARK_LIGHTING",
        "PARK_CLEANLINESS",
        "GRASS_MAINTENANCE",
        "PLANT_MAINTENANCE",
        "PUBLIC_GARDEN_DAMAGE",
        "TREE_BRANCH_HAZARD",
        "URBAN_GREENERY",
        "TREE_PLANTATION_REQUEST",
        "OTHER_GARDENS_GREENERY",
    ],
    "PUBLIC_HEALTH_SANITATION": [
        "MOSQUITO_BREEDING",
        "UNHYGIENIC_AREA",
        "PUBLIC_TOILET_ISSUE",
        "PUBLIC_TOILET_CLEANING",
        "DISEASE_RISK",
        "FOUL_ODOR",
        "STRAY_ANIMAL_ISSUE",
        "RODENT_INFESTATION",
        "PEST_CONTROL",
        "UNSANITARY_PUBLIC_SPACE",
        "HEALTH_HAZARD",
        "FOOD_HYGIENE_ISSUE",
        "WATERBORNE_HEALTH_RISK",
        "SANITATION_CLEANING",
        "OTHER_PUBLIC_HEALTH_SANITATION",
    ],
    "BUILDING_CONSTRUCTION": [
        "UNAUTHORIZED_CONSTRUCTION",
        "BUILDING_VIOLATION",
        "DANGEROUS_BUILDING",
        "STRUCTURAL_DAMAGE",
        "ILLEGAL_EXTENSION",
        "ILLEGAL_FLOOR_CONSTRUCTION",
        "CONSTRUCTION_WITHOUT_PERMISSION",
        "CONSTRUCTION_DEBRIS",
        "BUILDING_SAFETY_HAZARD",
        "WALL_COLLAPSE_RISK",
        "VACANT_DANGEROUS_STRUCTURE",
        "ENCLOSED_PUBLIC_SPACE",
        "CONSTRUCTION_NOISE",
        "BUILDING_INSPECTION_REQUEST",
        "OTHER_BUILDING_CONSTRUCTION",
    ],
    "ENCROACHMENT": [
        "FOOTPATH_ENCROACHMENT",
        "ROAD_ENCROACHMENT",
        "ILLEGAL_STALL",
        "ILLEGAL_SHOP_EXTENSION",
        "PUBLIC_LAND_ENCROACHMENT",
        "PARK_ENCROACHMENT",
        "ROAD_OBSTRUCTION",
        "FOOTPATH_OBSTRUCTION",
        "ILLEGAL_STRUCTURE",
        "TEMPORARY_ENCROACHMENT",
        "PERMANENT_ENCROACHMENT",
        "STREET_VENDOR_OBSTRUCTION",
        "PUBLIC_SPACE_OCCUPATION",
        "ACCESS_BLOCKAGE",
        "OTHER_ENCROACHMENT",
    ],
    "TRAFFIC_TRANSPORT_PARKING": [
        "ILLEGAL_PARKING",
        "PARKING_SHORTAGE",
        "TRAFFIC_OBSTRUCTION",
        "ROAD_BLOCKAGE",
        "BUS_STOP_ISSUE",
        "PUBLIC_TRANSPORT_ISSUE",
        "TRAFFIC_SIGNAL_ISSUE",
        "TRAFFIC_SIGN_DAMAGE",
        "ZEBRA_CROSSING_ISSUE",
        "PEDESTRIAN_SAFETY",
        "PARKING_ENFORCEMENT",
        "VEHICLE_OBSTRUCTION",
        "TRAFFIC_CONGESTION",
        "PUBLIC_TRANSPORT_STOP_ISSUE",
        "OTHER_TRAFFIC_TRANSPORT_PARKING",
    ],
    "PROPERTY_MUNICIPAL_TAX": [
        "PROPERTY_TAX_PAYMENT",
        "PROPERTY_TAX_ASSESSMENT",
        "PROPERTY_TAX_NOTICE",
        "PROPERTY_TAX_CORRECTION",
        "PROPERTY_TAX_REFUND",
        "PROPERTY_TAX_NAME_CHANGE",
        "PROPERTY_TAX_TRANSFER",
        "PROPERTY_TAX_DUPLICATE_BILL",
        "PROPERTY_TAX_DISPUTE",
        "PROPERTY_TAX_EXEMPTION",
        "PROPERTY_RECORD_CORRECTION",
        "PROPERTY_VALUATION",
        "TAX_RECEIPT_ISSUE",
        "TAX_ACCOUNT_ISSUE",
        "OTHER_PROPERTY_MUNICIPAL_TAX",
    ],
    "LICENSING_PERMISSIONS": [
        "TRADE_LICENSE",
        "LICENSE_RENEWAL",
        "LICENSE_APPLICATION",
        "LICENSE_STATUS",
        "LICENSE_CORRECTION",
        "SIGNAGE_PERMISSION",
        "HOARDING_PERMISSION",
        "PERMIT_APPLICATION",
        "PERMIT_RENEWAL",
        "PERMIT_STATUS",
        "BUSINESS_PERMISSION",
        "EVENT_PERMISSION",
        "STREET_VENDING_PERMISSION",
        "LICENSE_COMPLAINT",
        "OTHER_LICENSING_PERMISSIONS",
    ],
    "FIRE_DISASTER_EMERGENCY": [
        "FIRE_HAZARD",
        "FIRE_SAFETY_VIOLATION",
        "EMERGENCY_HAZARD",
        "BUILDING_FIRE_RISK",
        "BLOCKED_EMERGENCY_ACCESS",
        "DISASTER_HAZARD",
        "COLLAPSE_RISK",
        "ELECTRICAL_FIRE_RISK",
        "GAS_HAZARD",
        "FLOOD_EMERGENCY",
        "STORM_EMERGENCY",
        "PUBLIC_SAFETY_HAZARD",
        "EMERGENCY_INFRASTRUCTURE_DAMAGE",
        "DISASTER_RESPONSE_REQUEST",
        "OTHER_FIRE_DISASTER_EMERGENCY",
    ],
    "ENVIRONMENT_POLLUTION": [
        "AIR_POLLUTION",
        "NOISE_POLLUTION",
        "WATER_POLLUTION",
        "INDUSTRIAL_POLLUTION",
        "ILLEGAL_WASTE_DISPOSAL",
        "HAZARDOUS_WASTE",
        "DUST_POLLUTION",
        "SMOKE_POLLUTION",
        "CHEMICAL_CONTAMINATION",
        "POLLUTED_WATER_BODY",
        "ENVIRONMENTAL_VIOLATION",
        "ILLEGAL_DUMPING_ENVIRONMENT",
        "EXCESSIVE_NOISE",
        "POLLUTION_MONITORING_REQUEST",
        "OTHER_ENVIRONMENT_POLLUTION",
    ],
}

# ---------------------------------------------------------------------------
# Person 1 → Person 2 Domain Alias Bridge
# ---------------------------------------------------------------------------
# Person 1 (AI/NLP Pipeline) uses shortened domain names. Person 2 (Operational)
# uses fully qualified canonical names. This alias table allows transparent
# resolution so Person 1 output flows directly into Person 2 agents.

DOMAIN_ALIASES: Dict[str, str] = {
    "STREET_LIGHTING": "STREET_LIGHTING_ELECTRICAL",
    "GARDENS_TREES": "GARDENS_GREENERY",
    "PUBLIC_HEALTH": "PUBLIC_HEALTH_SANITATION",
    "TRAFFIC_PARKING": "TRAFFIC_TRANSPORT_PARKING",
    "PROPERTY_TAX": "PROPERTY_MUNICIPAL_TAX",
    "LICENSES_PERMISSIONS": "LICENSING_PERMISSIONS",
    "FIRE_DISASTER_SAFETY": "FIRE_DISASTER_EMERGENCY",
}

# Reverse lookup for debugging/reporting
REVERSE_ALIASES: Dict[str, str] = {v: k for k, v in DOMAIN_ALIASES.items()}


def normalize_domain_name(domain: str) -> str:
    """Normalizes Person 1 domain names to Person 2 canonical taxonomy names.
    
    If the domain is already a canonical Person 2 name, returns it unchanged.
    If it's a known Person 1 alias, returns the canonical equivalent.
    If unknown, returns the input unchanged (validation happens downstream).
    """
    if not domain:
        return domain
    return DOMAIN_ALIASES.get(domain, domain)


def all_domains() -> List[str]:
    return list(TAXONOMY.keys())


def all_subcategories(domain: str) -> List[str]:
    norm = normalize_domain_name(domain)
    return TAXONOMY.get(norm, [])


def is_valid_domain(domain: str) -> bool:
    norm = normalize_domain_name(domain)
    return norm in TAXONOMY


def is_valid_subcategory(domain: str, subcategory: str) -> bool:
    norm = normalize_domain_name(domain)
    return subcategory in TAXONOMY.get(norm, [])

