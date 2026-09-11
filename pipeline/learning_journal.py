"""
CivicFlow AI — Recursive Learning Journal
============================================

Records every pipeline execution outcome in a JSONL file for offline analysis.
Supports:
  - Recording predictions and their confidence
  - Recording human corrections (feedback loop)
  - Computing per-domain accuracy over time
  - Exporting accuracy reports for recalibration

The journal is append-only. Each line is a self-contained JSON record.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict


DEFAULT_JOURNAL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "learning_journal.jsonl"
)


class LearningJournal:
    """Append-only learning journal for recursive pipeline improvement."""

    def __init__(self, journal_path: Optional[str] = None):
        self.journal_path = Path(journal_path or DEFAULT_JOURNAL_PATH)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Recording
    # ──────────────────────────────────────────────────────────

    def record_outcome(
        self,
        complaint_text: str,
        complaint_id: Optional[str] = None,
        predicted_domain: Optional[str] = None,
        predicted_category: Optional[str] = None,
        confidence: float = 0.0,
        routing_department: Optional[str] = None,
        priority: Optional[str] = None,
        pipeline_status: str = "SUCCESS",
    ) -> Dict[str, Any]:
        """Record a pipeline prediction outcome."""
        record = {
            "type": "PREDICTION",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "complaint_id": complaint_id,
            "complaint_text_hash": hash(complaint_text) % (10**10),  # Privacy: hash only
            "complaint_text_preview": complaint_text[:80] + "..." if len(complaint_text) > 80 else complaint_text,
            "predicted_domain": predicted_domain,
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "routing_department": routing_department,
            "priority": priority,
            "pipeline_status": pipeline_status,
            "corrected": False,
            "correct_domain": None,
            "correct_category": None,
        }
        self._append(record)
        return record

    def record_correction(
        self,
        complaint_id: str,
        correct_domain: str,
        correct_category: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a human correction for a previous prediction."""
        record = {
            "type": "CORRECTION",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "complaint_id": complaint_id,
            "correct_domain": correct_domain,
            "correct_category": correct_category,
            "reason": reason,
        }
        self._append(record)
        return record

    # ──────────────────────────────────────────────────────────
    # Analysis & Reporting
    # ──────────────────────────────────────────────────────────

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Read all journal records."""
        if not self.journal_path.exists():
            return []
        records = []
        with open(self.journal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def get_predictions(self) -> List[Dict[str, Any]]:
        """Get only prediction records."""
        return [r for r in self.get_all_records() if r.get("type") == "PREDICTION"]

    def get_corrections(self) -> List[Dict[str, Any]]:
        """Get only correction records."""
        return [r for r in self.get_all_records() if r.get("type") == "CORRECTION"]

    def get_accuracy_report(self) -> Dict[str, Any]:
        """
        Compute accuracy statistics per domain using corrections as ground truth.

        Returns a dict with:
          - total_predictions: count of all predictions
          - total_corrections: count of corrections received
          - domain_accuracy: {domain: {predictions, correct, incorrect, accuracy}}
          - overall_accuracy: float (only over corrected predictions)
          - low_confidence_domains: list of domains with < 70% accuracy
          - confidence_calibration: {bucket: {count, correct_rate}}
        """
        predictions = self.get_predictions()
        corrections = self.get_corrections()

        # Build correction lookup by complaint_id
        correction_map: Dict[str, Dict] = {}
        for c in corrections:
            cid = c.get("complaint_id")
            if cid:
                correction_map[cid] = c

        # Per-domain stats
        domain_stats = defaultdict(lambda: {"predictions": 0, "correct": 0, "incorrect": 0})
        confidence_buckets = defaultdict(lambda: {"count": 0, "correct": 0})

        total_evaluated = 0
        total_correct = 0

        for pred in predictions:
            domain = pred.get("predicted_domain")
            if domain:
                domain_stats[domain]["predictions"] += 1

            cid = pred.get("complaint_id")
            conf = pred.get("confidence", 0.0)

            # Confidence bucket (0.0-0.2, 0.2-0.4, ...)
            bucket = f"{int(conf * 5) * 20}-{int(conf * 5) * 20 + 20}%"
            confidence_buckets[bucket]["count"] += 1

            if cid and cid in correction_map:
                total_evaluated += 1
                corr = correction_map[cid]
                if corr.get("correct_domain") == domain:
                    domain_stats[domain]["correct"] += 1
                    total_correct += 1
                    confidence_buckets[bucket]["correct"] += 1
                else:
                    domain_stats[domain]["incorrect"] += 1

        # Compute accuracy per domain
        domain_accuracy = {}
        low_confidence_domains = []
        for domain, stats in domain_stats.items():
            evaluated = stats["correct"] + stats["incorrect"]
            accuracy = stats["correct"] / evaluated if evaluated > 0 else None
            domain_accuracy[domain] = {
                "predictions": stats["predictions"],
                "evaluated": evaluated,
                "correct": stats["correct"],
                "incorrect": stats["incorrect"],
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
            }
            if accuracy is not None and accuracy < 0.70:
                low_confidence_domains.append(domain)

        # Confidence calibration
        calibration = {}
        for bucket, data in sorted(confidence_buckets.items()):
            rate = data["correct"] / data["count"] if data["count"] > 0 else None
            calibration[bucket] = {
                "count": data["count"],
                "correct_rate": round(rate, 4) if rate is not None else None,
            }

        overall_accuracy = total_correct / total_evaluated if total_evaluated > 0 else None

        return {
            "total_predictions": len(predictions),
            "total_corrections": len(corrections),
            "total_evaluated": total_evaluated,
            "overall_accuracy": round(overall_accuracy, 4) if overall_accuracy is not None else None,
            "domain_accuracy": domain_accuracy,
            "low_confidence_domains": low_confidence_domains,
            "confidence_calibration": calibration,
        }

    # ──────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────

    def _append(self, record: Dict[str, Any]) -> None:
        """Append a single JSON record to the journal file."""
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        """Clear the journal (for testing only)."""
        if self.journal_path.exists():
            self.journal_path.unlink()
