"""
Phase 4.2 Real Municipal Data Evaluation Engine.

Evaluates existing Phase 4 Baseline Model (Model A) on real-world held-out complaints (real_validation.csv),
performs error analysis, trains experimental real-data model (Model B) in ai/models/tfidf_real/,
and compares performance across real municipal data splits.
"""

import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List

# Ensure root workspace in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from ai.models.tfidf_model import HierarchicalTfidfModel
from ai.taxonomy.taxonomy_manager import TaxonomyManager
from ai.agents.understanding_agent import UnderstandingAgent


class RealDataEvaluator:
    """Evaluates Baseline vs Real-Trained models on real municipal data."""

    def __init__(self):
        self.tm = TaxonomyManager()
        self.understanding_agent = UnderstandingAgent()
        self.eval_results_dir = Path(__file__).parent / "results"
        self.real_models_dir = Path(__file__).parent.parent / "models" / "tfidf_real"
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)
        self.real_models_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model_on_real_data(self, model: HierarchicalTfidfModel, real_val_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates a model instance on real_validation.csv records."""
        y_true_cat = [r["category_code"] for r in real_val_records]
        y_true_dom = [r["domain_code"] for r in real_val_records]

        y_pred_cat = []
        y_pred_dom = []
        confidences = []

        errors = []
        confusions = Counter()

        for r in real_val_records:
            txt = r["complaint_text"]
            res = model.predict(txt)

            pred_dom = res["domain"]["code"]
            pred_cat = res["category"]["code"]
            conf = res["overall_confidence"]

            y_pred_dom.append(pred_dom)
            y_pred_cat.append(pred_cat)
            confidences.append(conf)

            if pred_cat != r["category_code"]:
                confusions[(r["category_code"], pred_cat)] += 1
                errors.append({
                    "complaint_id": r.get("complaint_id"),
                    "complaint_text": txt,
                    "true_domain": r["domain_code"],
                    "true_category": r["category_code"],
                    "pred_domain": pred_dom,
                    "pred_category": pred_cat,
                    "confidence": conf,
                    "source_file": r.get("source_file"),
                    "source_subcategory": r.get("source_subcategory")
                })

        dom_acc = accuracy_score(y_true_dom, y_pred_dom)
        cat_acc = accuracy_score(y_true_cat, y_pred_cat)
        cp, cr, cf1, _ = precision_recall_fscore_support(y_true_cat, y_pred_cat, average="macro", zero_division=0)
        cwp, cwr, cwf1, _ = precision_recall_fscore_support(y_true_cat, y_pred_cat, average="weighted", zero_division=0)

        # Per-domain metrics
        dp, dr, df1, _ = precision_recall_fscore_support(y_true_dom, y_pred_dom, average="macro", zero_division=0)

        top_confusions = [
            {"true": k[0], "pred": k[1], "count": v}
            for k, v in confusions.most_common(10)
        ]

        return {
            "total_samples": len(real_val_records),
            "domain_accuracy": round(float(dom_acc), 4),
            "category_accuracy": round(float(cat_acc), 4),
            "category_macro_f1": round(float(cf1), 4),
            "category_weighted_f1": round(float(cwf1), 4),
            "domain_macro_f1": round(float(df1), 4),
            "avg_confidence": round(float(sum(confidences) / max(1, len(confidences))), 4),
            "top_confusion_pairs": top_confusions,
            "errors": errors
        }

    def run_evaluation_pipeline(self) -> Dict[str, Any]:
        """Runs comparative evaluation between Baseline (Model A) and Real-Trained (Model B)."""
        real_val_path = Path(__file__).parent.parent / "data" / "real" / "real_validation.csv"
        real_train_path = Path(__file__).parent.parent / "data" / "real" / "real_train.csv"

        if not real_val_path.exists() or not real_train_path.exists():
            from ai.data.process_real_municipal_data import RealMunicipalDataProcessor
            proc = RealMunicipalDataProcessor()
            proc.process([
                r"c:\Users\Harsh Sakpal\Downloads\413fa9ec-8d06-4ecb-884e-1436c5a0f5dd.csv",
                r"c:\Users\Harsh Sakpal\Downloads\82f88d50-71c5-4203-92ac-5ccb5cabc7a2.csv",
                r"c:\Users\Harsh Sakpal\Downloads\9e7e6892-06b6-4fdc-967a-e4787562f155.csv",
                r"c:\Users\Harsh Sakpal\Downloads\b0d6e9ff-5eef-48bf-ba86-985dbe8112d1.csv",
                r"c:\Users\Harsh Sakpal\Downloads\b4dd8dd1-1628-4f35-9247-ef5afaad214d.csv",
                r"c:\Users\Harsh Sakpal\Downloads\d4419a76-e2af-44b3-aa25-369c85126f0f.csv",
            ])

        real_val_records = []
        with open(real_val_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                real_val_records.append(row)

        print(f"[Real Data Evaluation] Evaluating on {len(real_val_records)} held-out real validation samples...")

        # 1. Evaluate Existing Baseline Model (Phase 4 Synthetic TF-IDF)
        baseline_model = HierarchicalTfidfModel()
        baseline_model.load_artifacts()

        print("[Step 1] Evaluating Baseline Phase 4 Model A on Real Data...")
        baseline_results = self.evaluate_model_on_real_data(baseline_model, real_val_records)

        # Save Baseline Real Data Error Analysis
        error_report = {
            "eval_dataset": "real_validation.csv",
            "samples_count": len(real_val_records),
            "baseline_model_performance": {
                "domain_accuracy": baseline_results["domain_accuracy"],
                "category_accuracy": baseline_results["category_accuracy"],
                "category_macro_f1": baseline_results["category_macro_f1"],
                "category_weighted_f1": baseline_results["category_weighted_f1"],
                "avg_confidence": baseline_results["avg_confidence"],
            },
            "top_confusion_pairs": baseline_results["top_confusion_pairs"],
            "sample_errors": baseline_results["errors"][:15]
        }
        with open(self.eval_results_dir / "real_data_error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(error_report, f, indent=2)

        # 2. Train Experimental Real Data Model (Model B) in ai/models/tfidf_real/
        print("[Step 2] Training Experimental Real-Data Model B on real_train.csv...")
        real_model = HierarchicalTfidfModel(artifacts_dir=self.real_models_dir)
        real_model.train(str(real_train_path))

        print("[Step 3] Evaluating Real-Trained Model B on Real Validation Split...")
        real_model_results = self.evaluate_model_on_real_data(real_model, real_val_records)

        # Comparison Summary
        comparison_report = {
            "eval_samples": len(real_val_records),
            "baseline_phase4_model": {
                "domain_accuracy": baseline_results["domain_accuracy"],
                "category_accuracy": baseline_results["category_accuracy"],
                "macro_f1": baseline_results["category_macro_f1"],
                "weighted_f1": baseline_results["category_weighted_f1"],
                "avg_confidence": baseline_results["avg_confidence"]
            },
            "experimental_real_model": {
                "domain_accuracy": real_model_results["domain_accuracy"],
                "category_accuracy": real_model_results["category_accuracy"],
                "macro_f1": real_model_results["category_macro_f1"],
                "weighted_f1": real_model_results["category_weighted_f1"],
                "avg_confidence": real_model_results["avg_confidence"]
            },
            "performance_gain": {
                "domain_accuracy_diff": round(real_model_results["domain_accuracy"] - baseline_results["domain_accuracy"], 4),
                "category_accuracy_diff": round(real_model_results["category_accuracy"] - baseline_results["category_accuracy"], 4),
                "macro_f1_diff": round(real_model_results["category_macro_f1"] - baseline_results["category_macro_f1"], 4),
            }
        }

        with open(self.eval_results_dir / "real_model_comparison_report.json", "w", encoding="utf-8") as f:
            json.dump(comparison_report, f, indent=2)

        print("\n==================================================")
        print("REAL MUNICIPAL DATA EVALUATION REPORT")
        print("==================================================")
        print(f"Domain Accuracy:   Baseline = {baseline_results['domain_accuracy']:.2%} | Real-Trained = {real_model_results['domain_accuracy']:.2%}")
        print(f"Category Accuracy: Baseline = {baseline_results['category_accuracy']:.2%} | Real-Trained = {real_model_results['category_accuracy']:.2%}")
        print(f"Category Macro F1: Baseline = {baseline_results['category_macro_f1']:.4f} | Real-Trained = {real_model_results['category_macro_f1']:.4f}")
        print(f"Weighted F1:       Baseline = {baseline_results['category_weighted_f1']:.4f} | Real-Trained = {real_model_results['category_weighted_f1']:.4f}")
        print("==================================================")

        return comparison_report


if __name__ == "__main__":
    evaluator = RealDataEvaluator()
    evaluator.run_evaluation_pipeline()
