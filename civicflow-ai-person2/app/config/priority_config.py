"""
CivicFlow AI - Priority Agent Configuration
============================================

Configurable thresholds, feature weights, critical infrastructure lexicons,
and safety override rules for the DIPEX-style continuous priority engine.

NOTE: These values are configurable parameters tuned for CivicFlow AI and
do not claim to represent exact static constants from external packages.
"""

from typing import Dict

# ---------------------------------------------------------------------------
# 1. Feature Weights for Continuous Scoring (Sum = 1.0)
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS: Dict[str, float] = {
    "severity": 0.20,
    "public_impact": 0.12,
    "urgency": 0.15,
    "infrastructure_criticality": 0.15,
    "affected_population": 0.10,
    "recurrence": 0.05,
    "duration": 0.05,
    "sla_risk": 0.08,
    "hotspot": 0.05,
    "risk": 0.05,
}

# ---------------------------------------------------------------------------
# 2. Priority Score Thresholds
# ---------------------------------------------------------------------------
# score < low_max               -> LOW
# low_max <= score < medium_max -> MEDIUM
# medium_max <= score < high_max -> HIGH
# score >= high_max             -> CRITICAL
PRIORITY_THRESHOLDS: Dict[str, float] = {
    "low_max": 0.20,
    "medium_max": 0.40,
    "high_max": 0.60,
}

# ---------------------------------------------------------------------------
# 3. SLA Target Hours Mapping
# ---------------------------------------------------------------------------
SLA_TARGET_HOURS: Dict[str, int] = {
    "CRITICAL": 6,
    "HIGH": 24,
    "MEDIUM": 48,
    "LOW": 120,
}

# ---------------------------------------------------------------------------
# 4. Critical Infrastructure Keyword Signals
# ---------------------------------------------------------------------------
CRITICAL_INFRASTRUCTURE_PATTERNS = [
    r"\bwater main\b",
    r"\bmain pipeline\b",
    r"\bmajor road\b",
    r"\bhighway\b",
    r"\bexpressway\b",
    r"\belectrical grid\b",
    r"\btransformer\b",
    r"\bsubstation\b",
    r"\bpower station\b",
    r"\bsewage line\b",
    r"\bmain drain\b",
    r"\bemergency access\b",
    r"\bhospital access\b",
    r"\bfire station access\b",
    r"\bpublic safety infrastructure\b",
]

# ---------------------------------------------------------------------------
# 5. Deterministic Safety Override Patterns
# ---------------------------------------------------------------------------
# Directly forces CRITICAL priority with explicit override_applied = true
SAFETY_OVERRIDE_PATTERNS = [
    (r"\bfire hazard\b", "Fire Hazard / Explosion Risk"),
    (r"\bexplosion\b", "Fire Hazard / Explosion Risk"),
    (r"\blive wire\b", "Exposed High-Voltage Electrical Wire"),
    (r"\bexposed wire\b", "Exposed High-Voltage Electrical Wire"),
    (r"\belectrocution\b", "Immediate Electrical Risk"),
    (r"\bbuilding collapse\b", "Building Structure Collapse Risk"),
    (r"\bwall collapse\b", "Building Structure Collapse Risk"),
    (r"\btoxic chemical\b", "Major Chemical Contamination"),
    (r"\bchemical dump\b", "Major Chemical Contamination"),
    (r"\bpoisonous\b", "Major Contamination Risk"),
    (r"\bblocked emergency access\b", "Blocked Emergency Vehicle Access"),
    (r"\bmajor flood emergency\b", "Major Flood Emergency"),
]

# ---------------------------------------------------------------------------
# 6. Database Category Operational Defaults (Section 4 Context)
# ---------------------------------------------------------------------------
CATEGORY_CODE_BASELINES: Dict[str, str] = {
    "SAN-GARB": "MEDIUM",
    "SAN-DUMP": "HIGH",
    "RDW-POT": "HIGH",
    "RDW-WALK": "LOW",
    "ELE-LGT": "MEDIUM",
    "WTR-LEAK": "CRITICAL",
    "SWM-001": "MEDIUM",
    "SWM-002": "CRITICAL",
    "SWM-003": "LOW",
    "PHS-001": "HIGH",
    "PHS-002": "MEDIUM",
    "PHS-003": "HIGH",
    "PTX-001": "MEDIUM",
    "PTX-002": "MEDIUM",
    "LIC-001": "MEDIUM",
    "LIC-002": "LOW",
    "WS-001": "CRITICAL",
    "WS-002": "CRITICAL",
    "WS-003": "HIGH",
    "SD-001": "HIGH",
    "SD-002": "CRITICAL",
    "STW-001": "HIGH",
    "STW-002": "HIGH",
    "RDF-001": "HIGH",
    "RDF-002": "MEDIUM",
    "RDF-003": "MEDIUM",
    "BLD-001": "HIGH",
    "BLD-002": "CRITICAL",
    "ENC-001": "MEDIUM",
    "TRF-001": "MEDIUM",
    "TRF-002": "HIGH",
    "FDS-001": "CRITICAL",
    "FDS-002": "CRITICAL",
    "SLE-001": "HIGH",
    "SLE-002": "HIGH",
    "GTR-001": "HIGH",
    "GTR-002": "HIGH",
    "GTR-003": "LOW",
    "ENV-001": "MEDIUM",
    "ENV-002": "CRITICAL",
}
