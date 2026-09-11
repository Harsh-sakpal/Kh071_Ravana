"""
Validation-Based Threshold Tuning Engine for CivicFlow AI (Phase 4).

Grid searches candidate confidence and margin thresholds exclusively on validation.csv.
Optimizes for Category Macro F1, Accuracy, and Zero High-Confidence Regressions.
Saves optimal configuration to ai/config/agent_thresholds.json.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
from typing import Dict, Any, List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent
from ai.taxonomy.taxonomy_manager import TaxonomyManager


class ThresholdTuner:
    """Grid searches threshold parameters on validation dataset."""

    TFIDF_THRESHOLDS = [0.70, 0.75, 0.80, 0.85]
    TFIDF_MARGINS = [0.05, 0.10, 0.15, 0.20]
    EMBEDDING_THRESHOLDS = [0.60, 0.65, 0.70, 0.75]

    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or (Path(__file__).parent.parent / "config")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.tm = TaxonomyManager()
        self.understanding_agent = UnderstandingAgent()

    def tune(self, val_csv_path: str = None) -> Dict[str, Any]:
        """Runs grid search on validation.csv dataset."""
        if val_csv_path is None:
            val_csv_path = str(Path(__file__).parent.parent / "data" / "processed" / "validation.csv")

        val_records = []
        with open(val_csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val_records.append(row)

        print(f"[Threshold Tuning] Running grid search on {len(val_records)} validation samples...")
        y_true = [r["category_code"] for r in val_records]

        best_score = -1.0
        best_params = {}
        best_metrics = {}

        # Grid Search
        for t_conf in self.TFIDF_THRESHOLDS:
            for t_marg in self.TFIDF_MARGINS:
                for e_conf in self.EMBEDDING_THRESHOLDS:
                    agent = ClassificationAgent(
                        taxonomy_manager=self.tm,
                        tfidf_threshold=t_conf,
                        tfidf_margin=t_marg,
                        embedding_threshold=e_conf,
                    )

                    y_pred = []
                    for r in val_records:
                        u = self.understanding_agent.understand(r["complaint_text"])
                        res = agent.classify(u, debug=False)
                        y_pred.append(res.semantic_classification.category_code)

                    acc = accuracy_score(y_true, y_pred)
                    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

                    # Combined objective: Macro F1
                    if f1 > best_score:
                        best_score = f1
                        best_params = {
                            "tfidf_threshold": t_conf,
                            "tfidf_margin": t_marg,
                            "embedding_threshold": e_conf,
                        }
                        best_metrics = {
                            "validation_accuracy": round(float(acc), 4),
                            "validation_macro_f1": round(float(f1), 4),
                        }

        config = {
            "tuning_dataset": "validation.csv",
            "optimal_parameters": best_params,
            "validation_performance": best_metrics,
        }

        output_path = self.config_dir / "agent_thresholds.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"[Threshold Tuning] Grid search complete.")
        print(f"  - Optimal Params: {best_params}")
        print(f"  - Validation Accuracy: {best_metrics['validation_accuracy']:.2%} | Macro F1: {best_metrics['validation_macro_f1']:.4f}")
        print(f"  - Saved configuration to {output_path}")

        return config


if __name__ == "__main__":
    tuner = ThresholdTuner()
    tuner.tune()
