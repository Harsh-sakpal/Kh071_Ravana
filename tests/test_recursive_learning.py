"""
CivicFlow AI — Recursive Learning Tests
==========================================

Tests for the learning journal feedback loop:
  - Journal writes/reads correctly
  - Corrections update records
  - Accuracy report aggregation
  - Confidence recalibration
"""

import os
import sys
import json
import tempfile
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.learning_journal import LearningJournal
from pipeline.confidence_recalibrator import ConfidenceRecalibrator


@pytest.fixture
def temp_journal(tmp_path):
    """Create a LearningJournal with a temporary file path."""
    journal_path = str(tmp_path / "test_journal.jsonl")
    return LearningJournal(journal_path=journal_path)


@pytest.fixture
def temp_recalibrator(tmp_path):
    """Create a ConfidenceRecalibrator with temp paths."""
    journal_path = str(tmp_path / "test_journal.jsonl")
    cal_path = str(tmp_path / "test_calibration.json")
    journal = LearningJournal(journal_path=journal_path)
    return ConfidenceRecalibrator(journal=journal, calibration_path=cal_path)


# ──────────────────────────────────────────────────────────────────────────
# Learning Journal Tests
# ──────────────────────────────────────────────────────────────────────────

class TestLearningJournal:

    def test_record_outcome(self, temp_journal):
        record = temp_journal.record_outcome(
            complaint_text="Garbage not collected",
            complaint_id="CMP-001",
            predicted_domain="SOLID_WASTE",
            predicted_category="SAN-GARB",
            confidence=0.85,
            routing_department="SAN",
            priority="MEDIUM",
        )
        assert record["type"] == "PREDICTION"
        assert record["predicted_domain"] == "SOLID_WASTE"
        assert record["confidence"] == 0.85

    def test_records_persist_to_file(self, temp_journal):
        temp_journal.record_outcome(
            complaint_text="Test complaint 1",
            complaint_id="CMP-001",
            predicted_domain="WATER_SUPPLY",
            confidence=0.7,
        )
        temp_journal.record_outcome(
            complaint_text="Test complaint 2",
            complaint_id="CMP-002",
            predicted_domain="SOLID_WASTE",
            confidence=0.9,
        )
        records = temp_journal.get_all_records()
        assert len(records) == 2

    def test_record_correction(self, temp_journal):
        temp_journal.record_outcome(
            complaint_text="Streetlight issue",
            complaint_id="CMP-010",
            predicted_domain="ROADS_FOOTPATHS",
            confidence=0.4,
        )
        correction = temp_journal.record_correction(
            complaint_id="CMP-010",
            correct_domain="STREET_LIGHTING_ELECTRICAL",
            reason="Misclassified: was about streetlights, not roads",
        )
        assert correction["type"] == "CORRECTION"
        assert correction["correct_domain"] == "STREET_LIGHTING_ELECTRICAL"

    def test_get_predictions_and_corrections_separately(self, temp_journal):
        temp_journal.record_outcome(
            complaint_text="Test",
            complaint_id="CMP-001",
            predicted_domain="SOLID_WASTE",
            confidence=0.8,
        )
        temp_journal.record_correction(
            complaint_id="CMP-001",
            correct_domain="SOLID_WASTE",
        )
        assert len(temp_journal.get_predictions()) == 1
        assert len(temp_journal.get_corrections()) == 1

    def test_accuracy_report_with_corrections(self, temp_journal):
        # 3 predictions for SOLID_WASTE, 2 correct, 1 wrong
        for i in range(3):
            temp_journal.record_outcome(
                complaint_text=f"Garbage issue {i}",
                complaint_id=f"CMP-{i:03d}",
                predicted_domain="SOLID_WASTE",
                confidence=0.8,
            )

        temp_journal.record_correction("CMP-000", "SOLID_WASTE")
        temp_journal.record_correction("CMP-001", "SOLID_WASTE")
        temp_journal.record_correction("CMP-002", "WATER_SUPPLY")  # Wrong prediction

        report = temp_journal.get_accuracy_report()
        assert report["total_predictions"] == 3
        assert report["total_corrections"] == 3
        assert report["total_evaluated"] == 3

        sw_stats = report["domain_accuracy"]["SOLID_WASTE"]
        assert sw_stats["correct"] == 2
        assert sw_stats["incorrect"] == 1
        assert abs(sw_stats["accuracy"] - 0.6667) < 0.01

    def test_accuracy_report_empty_journal(self, temp_journal):
        report = temp_journal.get_accuracy_report()
        assert report["total_predictions"] == 0
        assert report["overall_accuracy"] is None

    def test_clear_journal(self, temp_journal):
        temp_journal.record_outcome(
            complaint_text="Test",
            predicted_domain="SOLID_WASTE",
            confidence=0.5,
        )
        assert len(temp_journal.get_all_records()) == 1
        temp_journal.clear()
        assert len(temp_journal.get_all_records()) == 0


# ──────────────────────────────────────────────────────────────────────────
# Confidence Recalibrator Tests
# ──────────────────────────────────────────────────────────────────────────

class TestConfidenceRecalibrator:

    def test_recalibrate_with_no_data(self, temp_recalibrator):
        cal = temp_recalibrator.recalibrate()
        assert cal["summary"]["total_predictions"] == 0
        assert len(cal["domain_adjustments"]) == 0

    def test_recalibrate_boosts_low_accuracy_domain(self, temp_recalibrator):
        journal = temp_recalibrator.journal

        # Create 5 predictions for ROADS_FOOTPATHS, only 1 correct → 20% accuracy
        for i in range(5):
            journal.record_outcome(
                complaint_text=f"Road issue {i}",
                complaint_id=f"ROAD-{i:03d}",
                predicted_domain="ROADS_FOOTPATHS",
                confidence=0.7,
            )
        journal.record_correction("ROAD-000", "ROADS_FOOTPATHS")  # Correct
        journal.record_correction("ROAD-001", "WATER_SUPPLY")     # Wrong
        journal.record_correction("ROAD-002", "SOLID_WASTE")      # Wrong
        journal.record_correction("ROAD-003", "WATER_SUPPLY")     # Wrong
        journal.record_correction("ROAD-004", "SOLID_WASTE")      # Wrong

        cal = temp_recalibrator.recalibrate()
        adj = cal["domain_adjustments"]["ROADS_FOOTPATHS"]
        assert adj["adjusted_floor"] > 0.25, "Floor should be boosted for low-accuracy domain"
        assert "LOW_ACCURACY" in adj["reason"]

    def test_recalibrate_no_boost_for_accurate_domain(self, temp_recalibrator):
        journal = temp_recalibrator.journal

        # Create 4 predictions for SOLID_WASTE, all correct → 100% accuracy
        for i in range(4):
            journal.record_outcome(
                complaint_text=f"Garbage issue {i}",
                complaint_id=f"SW-{i:03d}",
                predicted_domain="SOLID_WASTE",
                confidence=0.9,
            )
            journal.record_correction(f"SW-{i:03d}", "SOLID_WASTE")

        cal = temp_recalibrator.recalibrate()
        adj = cal["domain_adjustments"]["SOLID_WASTE"]
        assert adj["adjusted_floor"] == 0.25, "No boost needed for accurate domain"
        assert adj["reason"] == "ACCURACY_ACCEPTABLE"

    def test_load_and_get_adjusted_floor(self, temp_recalibrator):
        journal = temp_recalibrator.journal

        for i in range(4):
            journal.record_outcome(
                complaint_text=f"Test {i}",
                complaint_id=f"T-{i:03d}",
                predicted_domain="WATER_SUPPLY",
                confidence=0.6,
            )
            journal.record_correction(f"T-{i:03d}", "WATER_SUPPLY")

        temp_recalibrator.recalibrate()

        # Load and check
        floor = temp_recalibrator.get_adjusted_floor("WATER_SUPPLY")
        assert floor == 0.25  # 100% accuracy, no boost

    def test_calibration_persists_to_file(self, temp_recalibrator):
        temp_recalibrator.recalibrate()
        loaded = temp_recalibrator.load_calibration()
        assert loaded is not None
        assert "generated_at" in loaded
        assert "domain_adjustments" in loaded
