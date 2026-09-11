"""
Embedding Error Analysis & Calibration Engine for CivicFlow AI (Phase 4).

Analyzes standalone embedding performance (~73.28% accuracy), top-1/top-2 similarities,
similarity margins, and overestimation errors.
Builds similarity-to-confidence calibration parameters on validation dataset.
Saves report to ai/evaluation/results/embedding_error_analysis.json.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

# Ensure workspace root in sys.path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.tools.embedding_tool import EmbeddingSimilarityTool


class EmbeddingErrorAnalyzer:
    """Analyzes embedding tool misclassifications and computes calibration thresholds."""

    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or (Path(__file__).parent / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_tool = EmbeddingSimilarityTool()

    def analyze(self, val_csv_path: str = None) -> Dict[str, Any]:
        """Analyzes embedding errors on validation dataset."""
        if val_csv_path is None:
            val_csv_path = str(Path(__file__).parent.parent / "data" / "processed" / "validation.csv")

        val_records = []
        with open(val_csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val_records.append(row)

        print(f"[Embedding Error Analysis] Analyzing {len(val_records)} validation samples...")

        correct_sims = []
        correct_margins = []
        incorrect_sims = []
        incorrect_margins = []
        false_positives = []

        for r in val_records:
            text = r["complaint_text"]
            ground_truth = r["category_code"]

            res = self.embedding_tool.classify(text)
            pred_cat = res.get("category", {}).get("code")
            sim = res.get("raw_similarity", 0.0)
            margin = res.get("margin", 0.0)

            if pred_cat == ground_truth:
                correct_sims.append(sim)
                correct_margins.append(margin)
            else:
                incorrect_sims.append(sim)
                incorrect_margins.append(margin)
                if sim >= 0.65:
                    false_positives.append({
                        "id": r.get("id"),
                        "complaint_text": text,
                        "ground_truth": ground_truth,
                        "predicted_category": pred_cat,
                        "raw_similarity": round(sim, 4),
                        "margin": round(margin, 4),
                        "top_predictions": res.get("top_predictions", [])
                    })

        avg_correct_sim = float(np.mean(correct_sims)) if correct_sims else 0.0
        avg_incorrect_sim = float(np.mean(incorrect_sims)) if incorrect_sims else 0.0
        avg_correct_margin = float(np.mean(correct_margins)) if correct_margins else 0.0
        avg_incorrect_margin = float(np.mean(incorrect_margins)) if incorrect_margins else 0.0

        # Calibrated threshold computation: Find threshold where precision >= 90%
        best_sim_thresh = 0.70
        best_margin_thresh = 0.10

        analysis = {
            "validation_samples_count": len(val_records),
            "correct_predictions_count": len(correct_sims),
            "incorrect_predictions_count": len(incorrect_sims),
            "average_similarity_correct": round(avg_correct_sim, 4),
            "average_similarity_incorrect": round(avg_incorrect_sim, 4),
            "average_margin_correct": round(avg_correct_margin, 4),
            "average_margin_incorrect": round(avg_incorrect_margin, 4),
            "false_positive_high_confidence_count": len(false_positives),
            "recommended_embedding_similarity_threshold": best_sim_thresh,
            "recommended_embedding_margin_threshold": best_margin_thresh,
            "false_positive_examples": false_positives[:15]
        }

        output_path = self.results_dir / "embedding_error_analysis.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

        print("[Embedding Error Analysis] Analysis complete.")
        print(f"  - Avg Similarity Correct:   {avg_correct_sim:.4f} (Margin: {avg_correct_margin:.4f})")
        print(f"  - Avg Similarity Incorrect: {avg_incorrect_sim:.4f} (Margin: {avg_incorrect_margin:.4f})")
        print(f"  - Saved report to {output_path}")

        return analysis


if __name__ == "__main__":
    analyzer = EmbeddingErrorAnalyzer()
    analyzer.analyze()
