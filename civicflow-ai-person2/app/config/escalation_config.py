"""
CivicFlow AI - Advanced Escalation Agent Configuration
=========================================================

Thresholds, penalty multipliers, fallback officer workloads, department head contacts,
and escalation matrices for the Escalation Agent (ESCALATION_AGENT).
"""

from typing import Dict

# ---------------------------------------------------------------------------
# 1. Escalation Tier Threshold Ratios & Breach Probabilities
# ---------------------------------------------------------------------------
# TIER_0_NOMINAL: ratio < 0.75 and prob < 0.70
# TIER_1_SUPERVISOR_ALERT: ratio >= 0.75 or prob >= 0.70
# TIER_2_PRE_BREACH_REASSIGNMENT: ratio >= 0.90 or prob >= 0.85
# TIER_3_DEPARTMENT_HEAD_ESCALATION: ratio >= 1.00 (SLA Breached)
# TIER_4_EMERGENCY_OVERRIDE: Safety Override True or Affected Count >= 250 (when Breached)
ESCALATION_TIER_THRESHOLDS: Dict[str, float] = {
    "tier_1_ratio": 0.75,
    "tier_1_prob": 0.70,
    "tier_2_ratio": 0.90,
    "tier_2_prob": 0.85,
    "tier_3_ratio": 1.00,
    "tier_4_affected_threshold": 250,
}

# ---------------------------------------------------------------------------
# 2. Priority SLA Breach Penalty Multipliers (Points per Overtime Hour)
# ---------------------------------------------------------------------------
PRIORITY_PENALTY_WEIGHTS: Dict[str, float] = {
    "CRITICAL": 100.0,
    "HIGH": 50.0,
    "MEDIUM": 20.0,
    "LOW": 5.0,
}

# ---------------------------------------------------------------------------
# 3. Department Head Contacts (Migration V15 & Org Structure Defaults)
# ---------------------------------------------------------------------------
DEPARTMENT_HEAD_CONTACTS: Dict[str, Dict[str, str]] = {
    "SAN": {
        "title": "Head of Sanitation & Waste Management",
        "officer_id": "OFF-SAN-HEAD-01",
        "email": "head.sanitation@civicflow.gov",
        "phone": "+1-555-019-9001",
    },
    "RDW": {
        "title": "Chief Engineer of Roadworks & Infrastructure",
        "officer_id": "OFF-RDW-HEAD-01",
        "email": "head.roadworks@civicflow.gov",
        "phone": "+1-555-019-9002",
    },
    "WTR": {
        "title": "Director of Water Supply & Drainage Services",
        "officer_id": "OFF-WTR-HEAD-01",
        "email": "head.water@civicflow.gov",
        "phone": "+1-555-019-9003",
    },
    "ELE": {
        "title": "Superintendent of Electrical Grid & Lighting",
        "officer_id": "OFF-ELE-HEAD-01",
        "email": "head.electrical@civicflow.gov",
        "phone": "+1-555-019-9004",
    },
    "PRK": {
        "title": "Director of Parks & Environment",
        "officer_id": "OFF-PRK-HEAD-01",
        "email": "head.parks@civicflow.gov",
        "phone": "+1-555-019-9005",
    },
}
