"""
CivicFlow AI - Advanced SLA Monitoring Configuration
=====================================================

Configurable rules, category-specific SLA targets, subcategory complexity metrics,
alert tier thresholds, and 6-factor feature weights for the SLA Monitoring Agent.
"""

from typing import Dict

# ---------------------------------------------------------------------------
# 1. Feature Weights for Breach Probability Calculation (Sum = 1.0)
# ---------------------------------------------------------------------------
SLA_FEATURE_WEIGHTS: Dict[str, float] = {
    "time_elapsed": 0.35,
    "officer_workload": 0.20,
    "department_queue": 0.15,
    "complexity": 0.10,
    "location_urgency": 0.10,
    "historical_breach": 0.10,
}

# ---------------------------------------------------------------------------
# 2. Multi-Stage SLA Risk Tiers & Threshold Ratios
# ---------------------------------------------------------------------------
# elapsed_ratio < 0.50          -> ON_TRACK
# 0.50 <= ratio < 0.75          -> PROGRESSING
# 0.75 <= ratio < 0.90          -> NEAR_BREACH (Warning Tier 1)
# 0.90 <= ratio < 1.00          -> CRITICAL_RISK (Warning Tier 2 - Pre-Breach Alert)
# ratio >= 1.00                 -> BREACHED (Immediate Escalation Trigger)
SLA_TIER_THRESHOLDS: Dict[str, float] = {
    "progressing": 0.50,
    "near_breach": 0.75,
    "critical_risk": 0.90,
}

# ---------------------------------------------------------------------------
# 3. Category Code Custom SLA Durations (Hours)
# ---------------------------------------------------------------------------
# Derived from migration V15 seed data and domain rules
CATEGORY_SLA_DURATIONS: Dict[str, int] = {
    "WTR-LEAK": 6,
    "WS-001": 6,
    "WS-002": 6,
    "SD-002": 6,
    "SWM-002": 6,
    "BLD-002": 6,
    "FDS-001": 6,
    "FDS-002": 6,
    "ENV-002": 6,
    "SAN-DUMP": 24,
    "RDW-POT": 24,
    "PHS-001": 24,
    "PHS-003": 24,
    "WS-003": 24,
    "SD-001": 24,
    "STW-001": 24,
    "STW-002": 24,
    "RDF-001": 24,
    "BLD-001": 24,
    "TRF-002": 24,
    "SLE-001": 24,
    "SLE-002": 24,
    "GTR-001": 24,
    "GTR-002": 24,
    "SAN-GARB": 48,
    "ELE-LGT": 48,
    "PHS-002": 48,
    "PTX-001": 48,
    "PTX-002": 48,
    "LIC-001": 48,
    "RDF-002": 48,
    "RDF-003": 48,
    "ENC-001": 48,
    "TRF-001": 48,
    "ENV-001": 48,
    "RDW-WALK": 120,
    "SWM-003": 120,
    "LIC-002": 120,
    "GTR-003": 120,
}

# ---------------------------------------------------------------------------
# 4. Subcategory Work Complexity Scores (0.10 to 1.00)
# ---------------------------------------------------------------------------
SUBCATEGORY_COMPLEXITY: Dict[str, float] = {
    "WATER_MAIN_LEAK": 0.90,
    "OPEN_MANHOLE": 0.85,
    "STRUCTURAL_DAMAGE": 0.90,
    "DANGEROUS_BUILDING": 0.95,
    "EXPOSED_ELECTRICAL_WIRE": 0.85,
    "FIRE_HAZARD": 0.90,
    "CHEMICAL_CONTAMINATION": 0.85,
    "FLOODED_ROAD": 0.75,
    "LARGE_POTHOLE": 0.65,
    "GARBAGE_NOT_COLLECTED": 0.40,
    "STREETLIGHT_OUTAGE": 0.35,
    "PARK_MAINTENANCE": 0.20,
}
