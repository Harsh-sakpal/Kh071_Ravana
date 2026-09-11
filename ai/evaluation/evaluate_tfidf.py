"""
Evaluation & Error Analysis Engine for CivicFlow AI TF-IDF Baseline.

Evaluates Level 1 (Domain) and Level 2 (Category) classification quality on test dataset.
Computes Accuracy, Precision, Recall, Macro F1, Weighted F1, Top-3 Accuracy, and Confusion Analysis.
Saves evaluation metrics and error analysis reports to ai/evaluation/results/.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List

# Ensure workspace root in sys.path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from ai.models.tfidf_model import HierarchicalTfidfModel
from ai.taxonomy.taxonomy_manager import TaxonomyManager


class ModelEvaluator:
    """Evaluates hierarchical TF-IDF model on test dataset."""

    def __init__(self, model_dir: Path = None, results_dir: Path = None):
        self.model = HierarchicalTfidfModel(artifacts_dir=model_dir)
        self.results_dir = results_dir or (Path(__file__).parent / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tm = TaxonomyManager()

    def evaluate(self, test_csv_path: str) -> Dict[str, Any]:
        """Runs evaluation on test_csv_path dataset."""
        if not self.model.load_artifacts():
            raise RuntimeError("Failed to load model artifacts for evaluation.")

        test_records = []
        with open(test_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_records.append(row)

        print(f"[Evaluation] Loaded {len(test_records)} test samples from {test_csv_path}...")

        y_true_domain = []
        y_pred_domain = []

        y_true_cat = []
        y_pred_cat = []

        top3_correct = 0
        total_eval = 0

        errors = []
        confusions = defaultdict(int)

        for rec in test_records:
            text = rec["complaint_text"]
            true_dom = rec["domain_code"]
            true_cat = rec["category_code"]

            pred = self.model.predict(text, top_k=3)
            pred_dom = pred["domain"]["code"]
            pred_cat = pred["category"]["code"]
            top_preds = [p["code"] for p in pred.get("top_predictions", [])]

            y_true_domain.append(true_dom)
            y_pred_domain.append(pred_dom)

            y_true_cat.append(true_cat)
            y_pred_cat.append(pred_cat)

            if true_cat in top_preds:
                top3_correct += 1
            total_eval += 1

            # Track errors
            if pred_cat != true_cat:
                confusions[(true_cat, pred_cat)] += 1
                errors.append({
                    "id": rec.get("id"),
                    "complaint_text": text,
                    "true_domain": true_dom,
                    "predicted_domain": pred_dom,
                    "true_category": true_cat,
                    "predicted_category": pred_cat,
                    "overall_confidence": pred["overall_confidence"],
                    "top_predictions": pred.get("top_predictions", [])
                })

        # Calculate metrics
        dom_acc = accuracy_score(y_true_domain, y_pred_domain)
        dom_p, dom_r, dom_f1, _ = precision_recall_fscore_support(y_true_domain, y_pred_domain, average="macro", zero_division=0)
        dom_w_p, dom_w_r, dom_w_f1, _ = precision_recall_fscore_support(y_true_domain, y_pred_domain, average="weighted", zero_division=0)

        cat_acc = accuracy_score(y_true_cat, y_pred_cat)
        cat_p, cat_r, cat_f1, _ = precision_recall_fscore_support(y_true_cat, y_pred_cat, average="macro", zero_division=0)
        cat_w_p, cat_w_r, cat_w_f1, _ = precision_recall_fscore_support(y_true_cat, y_pred_cat, average="weighted", zero_division=0)

        top3_acc = top3_correct / max(total_eval, 1)

        # Format metrics report
        metrics = {
            "test_samples_count": len(test_records),
            "domain_metrics": {
                "accuracy": round(float(dom_acc), 4),
                "macro_precision": round(float(dom_p), 4),
                "macro_recall": round(float(dom_r), 4),
                "macro_f1": round(float(dom_f1), 4),
                "weighted_f1": round(float(dom_w_f1), 4),
            },
            "category_metrics": {
                "accuracy": round(float(cat_acc), 4),
                "top3_accuracy": round(float(top3_acc), 4),
                "macro_precision": round(float(cat_p), 4),
                "macro_recall": round(float(cat_r), 4),
                "macro_f1": round(float(cat_f1), 4),
                "weighted_f1": round(float(cat_w_f1), 4),
            }
        }

        # Save machine-readable evaluation metrics
        with open(self.results_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Format error analysis
        top_confusions = sorted(
            [{"true_category": k[0], "predicted_category": k[1], "count": v} for k, v in confusions.items()],
            key=lambda x: x["count"],
            reverse=True
        )

        error_report = {
            "total_errors": len(errors),
            "error_rate": round(len(errors) / max(total_eval, 1), 4),
            "top_category_confusions": top_confusions[:15],
            "misclassified_samples": errors[:30]
        }

        with open(self.results_dir / "error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(error_report, f, indent=2)

        # Write human readable markdown report
        with open(self.results_dir / "human_readable_report.md", "w", encoding="utf-8") as f:
            f.write("# CivicFlow AI - Phase 2 Evaluation & Error Analysis Report\n\n")
            f.write(f"**Test Set Samples**: {len(test_records)}\n\n")
            f.write("## 1. Domain Level Classification (15 AI Domains)\n")
            f.write(f"- **Accuracy**: `{metrics['domain_metrics']['accuracy']:.2%}`\n")
            f.write(f"- **Macro F1 Score**: `{metrics['domain_metrics']['macro_f1']:.4f}`\n")
            f.write(f"- **Weighted F1 Score**: `{metrics['domain_metrics']['weighted_f1']:.4f}`\n\n")

            f.write("## 2. Category Level Classification (34 Categories)\n")
            f.write(f"- **Top-1 Accuracy**: `{metrics['category_metrics']['accuracy']:.2%}`\n")
            f.write(f"- **Top-3 Accuracy**: `{metrics['category_metrics']['top3_accuracy']:.2%}`\n")
            f.write(f"- **Macro F1 Score**: `{metrics['category_metrics']['macro_f1']:.4f}`\n")
            f.write(f"- **Weighted F1 Score**: `{metrics['category_metrics']['weighted_f1']:.4f}`\n\n")

            f.write("## 3. Top Category Confusion Pairs\n\n")
            f.write("| True Category | Predicted Category | Frequency |\n")
            f.write("|--------------|-------------------|----------|\n")
            for c in top_confusions[:10]:
                f.write(f"| `{c['true_category']}` | `{c['predicted_category']}` | {c['count']} |\n")

        print("[Evaluation] Evaluation complete. Saved results to ai/evaluation/results/")
        print(f"  - Domain Accuracy:   {metrics['domain_metrics']['accuracy']:.2%}")
        print(f"  - Category Accuracy: {metrics['category_metrics']['accuracy']:.2%}")
        print(f"  - Top-3 Accuracy:    {metrics['category_metrics']['top3_accuracy']:.2%}")
        print(f"  - Category Macro F1: {metrics['category_metrics']['macro_f1']:.4f}")

        return metrics


if __name__ == "__main__":
    test_csv = Path(__file__).parent.parent / "data" / "processed" / "test.csv"
    evaluator = ModelEvaluator()
    evaluator.evaluate(str(test_csv))
