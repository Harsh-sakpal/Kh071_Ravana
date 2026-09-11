"""
CivicFlow AI - Routing Keyword Lexicon
========================================

Rule/semantic-signal layer for the Routing Agent's hybrid pipeline
(Master Prompt Section 9):

    Complaint -> Normalization -> Rules/semantic signals -> AI/NLP
    reasoning where required -> Candidate classification -> Validation
    -> Final routing result

This module supplies the "rules / semantic signals" stage: a keyword
lexicon per (domain, subcategory) used for scoring. It is intentionally
config-driven and separate from the agent's logic (Master Prompt Section
18) so weights/terms can be tuned without touching routing_agent.py.

Coverage note: not all 225 subcategories have hand-tuned keyword lists yet
(MVP-first, per Section 19 #15). Uncovered subcategories fall back to
DOMAIN_KEYWORDS for domain-level detection; the agent will surface low
confidence and, where appropriate, the domain's OTHER_* subcategory.
Extend SUBCATEGORY_KEYWORDS incrementally as real complaint data comes in.
"""

from typing import Dict, List, Tuple

# Domain-level keywords: used to pick the most likely domain before
# narrowing to a subcategory within it.
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "SOLID_WASTE": ["garbage", "trash", "waste", "dump", "dumping", "litter", "bin overflow", "rubbish"],
    "WATER_SUPPLY": ["water supply", "no water", "water pressure", "pipeline", "water pipe", "tanker", "water meter"],
    "SEWERAGE_DRAINAGE": ["sewage", "sewer", "manhole", "drain block", "drainage smell", "backflow"],
    "STORM_WATER": ["waterlogging", "flooded road", "storm drain", "rainwater", "monsoon flooding", "culvert"],
    "ROADS_FOOTPATHS": ["pothole", "road damage", "footpath", "sidewalk", "road crack", "speed breaker", "road cave"],
    "STREET_LIGHTING_ELECTRICAL": ["streetlight", "street light", "dark street", "light pole", "exposed wire", "sparking", "electric pole"],
    "GARDENS_GREENERY": ["park", "tree", "fallen tree", "garden", "pruning", "playground equipment", "grass"],
    "PUBLIC_HEALTH_SANITATION": ["mosquito", "stray dog", "public toilet", "foul smell", "rodent", "pest", "unhygienic"],
    "BUILDING_CONSTRUCTION": ["illegal construction", "unauthorized building", "dilapidated building", "structural damage", "building collapse"],
    "ENCROACHMENT": ["encroachment", "illegal stall", "footpath blocked", "hawkers", "vendor obstruction", "land grab"],
    "TRAFFIC_TRANSPORT_PARKING": ["illegal parking", "traffic signal", "traffic jam", "bus stop", "zebra crossing", "abandoned vehicle"],
    "PROPERTY_MUNICIPAL_TAX": ["property tax", "tax assessment", "tax bill", "tax refund", "tax receipt"],
    "LICENSING_PERMISSIONS": ["trade license", "permit", "hoarding permission", "shop license", "vending license"],
    "FIRE_DISASTER_EMERGENCY": ["fire hazard", "gas leak", "gas hazard", "collapse risk", "emergency access blocked", "disaster"],
    "ENVIRONMENT_POLLUTION": ["air pollution", "noise pollution", "industrial pollution", "chemical dumping", "smoke", "dust pollution"],
}

# Subcategory-level keywords: (domain, subcategory) -> keyword list.
# Weighted more heavily than domain-level matches during scoring.
SUBCATEGORY_KEYWORDS: Dict[Tuple[str, str], List[str]] = {
    ("SOLID_WASTE", "GARBAGE_NOT_COLLECTED"): ["not collected", "not been collected", "no collection", "days garbage", "piling up", "not picked up"],
    ("SOLID_WASTE", "MISSED_GARBAGE_COLLECTION"): ["missed collection", "skipped pickup", "truck did not come"],
    ("SOLID_WASTE", "OVERFLOWING_GARBAGE_BIN"): ["overflowing bin", "bin full", "garbage bin overflow"],
    ("SOLID_WASTE", "ILLEGAL_DUMPING"): ["illegal dumping", "dumped waste", "dumping ground", "unauthorized dumping"],
    ("SOLID_WASTE", "DEAD_ANIMAL_WASTE"): ["dead animal", "animal carcass", "dead dog", "dead cow"],

    ("WATER_SUPPLY", "NO_WATER_SUPPLY"): ["no water", "water stopped", "no supply of water"],
    ("WATER_SUPPLY", "WATER_MAIN_LEAK"): ["main pipe burst", "water main leak", "pipeline burst", "major leak"],
    ("WATER_SUPPLY", "PIPELINE_LEAK"): ["pipe leaking", "pipeline leak", "leaking pipe"],
    ("WATER_SUPPLY", "DIRTY_WATER"): ["dirty water", "contaminated water", "muddy water"],
    ("WATER_SUPPLY", "LOW_WATER_PRESSURE"): ["low pressure", "weak water flow"],

    ("SEWERAGE_DRAINAGE", "SEWAGE_OVERFLOW"): ["sewage overflow", "sewage on street", "sewage flooding"],
    ("SEWERAGE_DRAINAGE", "OPEN_MANHOLE"): ["open manhole", "manhole cover missing", "uncovered manhole"],
    ("SEWERAGE_DRAINAGE", "BLOCKED_SEWER"): ["blocked sewer", "sewer clogged", "sewer blockage"],

    ("STORM_WATER", "WATERLOGGING"): ["waterlogging", "water logged", "flooded street"],
    ("STORM_WATER", "BLOCKED_STORM_DRAIN"): ["storm drain blocked", "drain choked"],

    ("ROADS_FOOTPATHS", "POTHOLE"): ["pothole", "pot hole", "hole in the road"],
    ("ROADS_FOOTPATHS", "DAMAGED_ROAD"): ["road damaged", "broken road", "road surface damage"],
    ("ROADS_FOOTPATHS", "BROKEN_FOOTPATH"): ["broken footpath", "damaged sidewalk", "footpath broken"],

    ("STREET_LIGHTING_ELECTRICAL", "STREETLIGHT_OUTAGE"): ["streetlight not working", "light outage", "streetlight off"],
    ("STREET_LIGHTING_ELECTRICAL", "EXPOSED_ELECTRICAL_WIRE"): ["exposed wire", "live wire", "bare wire hanging"],
    ("STREET_LIGHTING_ELECTRICAL", "DARK_STREET"): ["dark street", "no light at night", "street is dark"],

    ("GARDENS_GREENERY", "FALLEN_TREE"): ["fallen tree", "tree fell", "tree collapsed"],
    ("GARDENS_GREENERY", "TREE_BRANCH_HAZARD"): ["hanging branch", "branch about to fall", "dangerous branch"],
    ("GARDENS_GREENERY", "PARK_MAINTENANCE"): ["park unmaintained", "park in bad condition"],

    ("PUBLIC_HEALTH_SANITATION", "MOSQUITO_BREEDING"): ["mosquito breeding", "stagnant water mosquito", "dengue risk"],
    ("PUBLIC_HEALTH_SANITATION", "STRAY_ANIMAL_ISSUE"): ["stray dogs", "aggressive dogs", "stray cattle"],
    ("PUBLIC_HEALTH_SANITATION", "PUBLIC_TOILET_ISSUE"): ["public toilet", "washroom dirty", "toilet broken"],

    ("BUILDING_CONSTRUCTION", "UNAUTHORIZED_CONSTRUCTION"): ["illegal construction", "unauthorized building", "construction without permission"],
    ("BUILDING_CONSTRUCTION", "DANGEROUS_BUILDING"): ["building about to collapse", "dangerous building", "dilapidated structure"],

    ("ENCROACHMENT", "FOOTPATH_ENCROACHMENT"): ["footpath encroachment", "hawkers on footpath", "stall blocking footpath"],

    ("TRAFFIC_TRANSPORT_PARKING", "ILLEGAL_PARKING"): ["illegal parking", "double parked", "vehicle blocking road"],
    ("TRAFFIC_TRANSPORT_PARKING", "TRAFFIC_SIGNAL_ISSUE"): ["traffic signal not working", "traffic light broken"],
    ("TRAFFIC_TRANSPORT_PARKING", "ABANDONED_VEHICLE"): ["abandoned vehicle", "rotting car", "junk vehicle"],

    ("PROPERTY_MUNICIPAL_TAX", "PROPERTY_TAX_ASSESSMENT"): ["tax assessment wrong", "incorrect property tax"],
    ("PROPERTY_MUNICIPAL_TAX", "PROPERTY_TAX_PAYMENT"): ["tax payment failed", "payment deducted no receipt"],

    ("LICENSING_PERMISSIONS", "TRADE_LICENSE"): ["trade license", "business license application"],
    ("LICENSING_PERMISSIONS", "HOARDING_PERMISSION"): ["illegal hoarding", "unauthorized banner", "hoarding permission"],

    ("FIRE_DISASTER_EMERGENCY", "FIRE_HAZARD"): ["fire hazard", "risk of fire", "flammable material stored"],
    ("FIRE_DISASTER_EMERGENCY", "GAS_HAZARD"): ["gas leak", "gas smell", "gas hazard"],

    ("ENVIRONMENT_POLLUTION", "AIR_POLLUTION"): ["air pollution", "smoke in air", "poor air quality"],
    ("ENVIRONMENT_POLLUTION", "NOISE_POLLUTION"): ["noise pollution", "loud noise complaint", "excessive noise"],
    ("ENVIRONMENT_POLLUTION", "CHEMICAL_CONTAMINATION"): ["chemical dumping", "toxic waste", "factory chemical dumping"],
}
