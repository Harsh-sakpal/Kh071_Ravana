"""
CivicFlow AI — Confidence Recalibrator
=========================================

Reads the learning journal and adjusts per-domain confidence thresholds
based on historical hit rate. This is the "recursive improvement" loop:

    Pipeline runs → Journal records → Recalibrator analyzes →
    Thresholds updated → Next pipeline run uses updated thresholds

The recalibrator does NOT modify agent code. Instead, it produces a
calibration config that the orchestrator can optionally consume.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from pipeline.learning_journal import LearningJournal


DEFAULT_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "calibration.json"
)

# Default confidence floor (matches Person 2 RoutingAgent's CONFIDENCE_LOW_FLOOR)
DEFAULT_CONFIDENCE_FLOOR = 0.25

# If a domain's accuracy drops below this, its confidence floor is raised
ACCURACY_CONCERN_THRESHOLD = 0.70

# Maximum confidence floor adjustment
MAX_FLOOR_BOOST = 0.20


class ConfidenceRecalibrator:
    """
    Analyzes learning journal data and produces per-domain confidence
    adjustments based on historical accuracy.
    """

    def __init__(
        self,
        journal: Optional[LearningJournal] = None,
        calibration_path: Optional[str] = None,
    ):
        self.journal = journal or LearningJournal()
        self.calibration_path = Path(calibration_path or DEFAULT_CALIBRATION_PATH)
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)

    def recalibrate(self) -> Dict:
        """
        Run recalibration analysis and persist updated thresholds.

        Returns the calibration config dict.
        """
        report = self.journal.get_accuracy_report()
        domain_accuracy = report.get("domain_accuracy", {})

        calibration = {
            "generated_at": None,
            "base_confidence_floor": DEFAULT_CONFIDENCE_FLOOR,
            "domain_adjustments": {},
            "low_confidence_domains": report.get("low_confidence_domains", []),
            "summary": {},
        }

        from datetime import datetime, timezone
        calibration["generated_at"] = datetime.now(timezone.utc).isoformat()

        adjustments = {}
        for domain, stats in domain_accuracy.items():
            accuracy = stats.get("accuracy")
            evaluated = stats.get("evaluated", 0)

            if accuracy is None or evaluated < 3:
                # Not enough data to recalibrate
                adjustments[domain] = {
                    "adjusted_floor": DEFAULT_CONFIDENCE_FLOOR,
                    "reason": "INSUFFICIENT_DATA",
                    "samples": evaluated,
                }
                continue

            if accuracy < ACCURACY_CONCERN_THRESHOLD:
                # Raise confidence floor proportionally to inaccuracy
                inaccuracy_ratio = 1.0 - accuracy
                boost = min(MAX_FLOOR_BOOST, inaccuracy_ratio * 0.5)
                new_floor = round(DEFAULT_CONFIDENCE_FLOOR + boost, 4)
                adjustments[domain] = {
                    "adjusted_floor": new_floor,
                    "reason": f"LOW_ACCURACY ({accuracy:.1%})",
                    "accuracy": accuracy,
                    "samples": evaluated,
                    "boost_applied": round(boost, 4),
                }
            else:
                adjustments[domain] = {
                    "adjusted_floor": DEFAULT_CONFIDENCE_FLOOR,
                    "reason": "ACCURACY_ACCEPTABLE",
                    "accuracy": accuracy,
                    "samples": evaluated,
                }

        calibration["domain_adjustments"] = adjustments
        calibration["summary"] = {
            "total_predictions": report.get("total_predictions", 0),
            "total_corrections": report.get("total_corrections", 0),
            "overall_accuracy": report.get("overall_accuracy"),
            "domains_requiring_boost": len(report.get("low_confidence_domains", [])),
        }

        # Persist
        with open(self.calibration_path, "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2, ensure_ascii=False)

        return calibration

    def load_calibration(self) -> Optional[Dict]:
        """Load the most recent calibration config."""
        if not self.calibration_path.exists():
            return None
        with open(self.calibration_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_adjusted_floor(self, domain: str) -> float:
        """Get the recalibrated confidence floor for a domain."""
        cal = self.load_calibration()
        if not cal:
            return DEFAULT_CONFIDENCE_FLOOR
        adj = cal.get("domain_adjustments", {}).get(domain, {})
        return adj.get("adjusted_floor", DEFAULT_CONFIDENCE_FLOOR)
