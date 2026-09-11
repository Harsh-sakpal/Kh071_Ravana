"""
Agent Regression Analysis Engine for CivicFlow AI (Phase 4).

Analyzes every challenge-set sample where TF-IDF prediction was correct BUT the Agentic
Pipeline escalated and produced an incorrect final category prediction.
Quantifies TF-IDF -> Embedding and TF-IDF -> LLM regressions.
Saves detailed report to ai/evaluation/results/agent_regression_analysis.json.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure workspace root in sys.path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent
from ai.tools.tfidf_tool import TfidfClassificationTool
from ai.tools.embedding_tool import EmbeddingSimilarityTool


class RegressionAnalyzer:
    """Analyzes cases where secondary tool escalation degraded accuracy."""

    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or (Path(__file__).parent / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.tfidf_tool = TfidfClassificationTool()
        try:
            self.embedding_tool = EmbeddingSimilarityTool()
        except Exception:
            self.embedding_tool = None

        self.understanding_agent = UnderstandingAgent()
        self.classification_agent = ClassificationAgent()

    def analyze(self, challenge_csv_path: str = None) -> Dict[str, Any]:
        """Runs regression analysis on challenge dataset."""
        if challenge_csv_path is None:
            challenge_csv_path = str(Path(__file__).parent.parent / "data" / "processed" / "challenge_test.csv")

        test_records = []
        with open(challenge_csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                test_records.append(row)

        print(f"[Regression Analysis] Analyzing {len(test_records)} challenge samples...")

        regressions = []
        tfidf_to_emb_count = 0
        tfidf_to_llm_count = 0
        emb_to_llm_count = 0

        for r in test_records:
            text = r["complaint_text"]
            ground_truth = r["category_code"]

            # Get TF-IDF standalone prediction
            tfidf_res = self.tfidf_tool.classify(text)
            tfidf_cat = tfidf_res.get("category", {}).get("code")
            tfidf_conf = tfidf_res.get("overall_confidence", 0.0)
            tfidf_preds = tfidf_res.get("top_predictions", [])
            tfidf_margin = 0.0
            if len(tfidf_preds) > 1:
                tfidf_margin = tfidf_preds[0]["score"] - tfidf_preds[1]["score"]

            # Get Embedding standalone prediction if available
            emb_res = self.embedding_tool.classify(text) if (self.embedding_tool and self.embedding_tool.is_ready) else {}
            emb_cat = emb_res.get("category", {}).get("code")
            emb_conf = emb_res.get("overall_confidence", 0.0)

            # Get Agent pipeline prediction
            understanding = self.understanding_agent.understand(text)
            agent_res = self.classification_agent.classify(understanding, debug=True)
            agent_cat = agent_res.semantic_classification.category_code
            agent_method = agent_res.method

            trace = agent_res.secondary_signals[0].get("trace", []) if agent_res.secondary_signals else []

            # Check if TF-IDF was correct BUT Agent prediction was incorrect
            is_tfidf_correct = (tfidf_cat == ground_truth)
            is_agent_correct = (agent_cat == ground_truth)

            if is_tfidf_correct and not is_agent_correct:
                if agent_method == "EMBEDDING":
                    tfidf_to_emb_count += 1
                    reason = "TF-IDF escalated to Embedding due to conservative threshold; Embedding misclassified."
                elif agent_method == "LLM_FALLBACK":
                    tfidf_to_llm_count += 1
                    reason = "TF-IDF escalated past Embedding to LLM Fallback; LLM misclassified."
                else:
                    reason = f"TF-IDF escalated to {agent_method}; final result misclassified."

                regressions.append({
                    "id": r.get("id"),
                    "complaint_text": text,
                    "ground_truth": ground_truth,
                    "tfidf_prediction": tfidf_cat,
                    "tfidf_confidence": round(tfidf_conf, 4),
                    "tfidf_margin": round(tfidf_margin, 4),
                    "embedding_prediction": emb_cat,
                    "embedding_confidence": round(emb_conf, 4),
                    "final_agent_prediction": agent_cat,
                    "agent_method": agent_method,
                    "tool_trace": trace,
                    "reason_for_escalation": "TF-IDF confidence or margin below threshold",
                    "reason_final_became_incorrect": reason
                })

        report = {
            "total_challenge_samples": len(test_records),
            "total_regressions": len(regressions),
            "regression_rate": round(len(regressions) / max(1, len(test_records)), 4),
            "tfidf_to_embedding_regressions": tfidf_to_emb_count,
            "tfidf_to_llm_regressions": tfidf_to_llm_count,
            "embedding_to_llm_regressions": emb_to_llm_count,
            "regression_records": regressions
        }

        output_path = self.results_dir / "agent_regression_analysis.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"[Regression Analysis] Analysis complete.")
        print(f"  - Total Regressions (TF-IDF correct -> Agent incorrect): {len(regressions)}")
        print(f"  - TF-IDF -> Embedding Regressions: {tfidf_to_emb_count}")
        print(f"  - TF-IDF -> LLM Regressions:       {tfidf_to_llm_count}")
        print(f"  - Saved report to {output_path}")

        return report


if __name__ == "__main__":
    analyzer = RegressionAnalyzer()
    analyzer.analyze()
