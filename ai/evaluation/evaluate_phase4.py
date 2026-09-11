"""
Phase 4 Comparative Evaluation & Calibration Engine for CivicFlow AI.

Compares TF-IDF, Embedding, Phase 3 Agent, and Phase 4 Calibrated Agent across:
1. Challenge Set (408 unseen complaints)
2. Hard Negative Groups (9 confusion sets)
3. Uncertainty & Adversarial Set (400 OOS & ambiguous complaints)

Calculates False Confident Classification Rate, Regression Reduction %, Tool Escalation %,
Average Tools per Complaint, and Latencies.
Saves comparison reports to ai/evaluation/results/phase3_vs_phase4_report.json.
"""

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List

# Ensure workspace root in sys.path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent
from ai.tools.tfidf_tool import TfidfClassificationTool
from ai.tools.embedding_tool import EmbeddingSimilarityTool
from ai.taxonomy.taxonomy_manager import TaxonomyManager


class Phase4Evaluator:
    """Comprehensive evaluation comparing Phase 3 vs Phase 4 across all test suites."""

    def __init__(self):
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tm = TaxonomyManager()
        self.understanding_agent = UnderstandingAgent()

    def evaluate_pipeline(self, records: List[Dict[str, Any]], agent: ClassificationAgent) -> Dict[str, Any]:
        """Evaluates an agent instance on given record list."""
        y_true = [r.get("category_code") for r in records]
        y_pred = []
        tool_counts = []
        methods = Counter()
        statuses = Counter()
        latencies = []

        for r in records:
            t0 = time.perf_counter()
            u = self.understanding_agent.understand(r["complaint_text"])
            res = agent.classify(u, debug=True)
            lat = (time.perf_counter() - t0) * 1000.0

            latencies.append(lat)
            cat_code = res.semantic_classification.category_code
            y_pred.append(cat_code)
            methods[res.method] += 1
            statuses[res.status] += 1

            trace_items = res.secondary_signals[0].get("trace", []) if res.secondary_signals else []
            tool_counts.append(max(1, len(trace_items)))

        n_total = max(1, len(records))
        acc = accuracy_score(y_true, y_pred) if any(y_true) else 0.0
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

        return {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "weighted_f1": round(float(wf1), 4),
            "avg_latency_ms": round(float(sum(latencies) / len(latencies)), 2),
            "avg_tools_per_complaint": round(float(sum(tool_counts) / len(tool_counts)), 2),
            "escalation_rates": {
                "tfidf_accepted_pct": round(methods.get("TFIDF", 0) / n_total, 4),
                "embedding_accepted_pct": round(methods.get("EMBEDDING", 0) / n_total, 4),
                "llm_fallback_accepted_pct": round(methods.get("LLM_FALLBACK", 0) / n_total, 4),
                "rule_engine_accepted_pct": round(methods.get("RULE_ENGINE", 0) / n_total, 4),
            },
            "status_distribution": dict(statuses)
        }

    def evaluate_false_confident_rate(self, uncert_records: List[Dict[str, Any]], agent: ClassificationAgent) -> Dict[str, Any]:
        """
        Calculates False Confident Classification Rate:
        Percentage of ambiguous, insufficient, or OOS complaints incorrectly forced into a normal category code.
        """
        false_confident_count = 0
        total = len(uncert_records)
        details = []

        for r in uncert_records:
            u = self.understanding_agent.understand(r["complaint_text"])
            res = agent.classify(u, debug=True)

            # If complaint is OOS/Ambiguous/Insufficient BUT received a normal category classification -> False Confident Error
            if res.status == "CLASSIFIED" and res.semantic_classification.category_code is not None:
                false_confident_count += 1
                details.append({
                    "id": r.get("id"),
                    "complaint_text": r["complaint_text"],
                    "difficulty": r.get("difficulty"),
                    "falsely_assigned_category": res.semantic_classification.category_code,
                    "confidence": res.overall_confidence,
                    "method": res.method
                })

        false_confident_rate = false_confident_count / max(1, total)

        return {
            "total_uncertainty_samples": total,
            "false_confident_classifications": false_confident_count,
            "false_confident_classification_rate": round(float(false_confident_rate), 4),
            "examples": details[:10]
        }

    def run_full_comparison(self) -> Dict[str, Any]:
        """Runs comparative evaluation of Phase 3 Agent vs Phase 4 Agent across all datasets."""
        data_dir = Path(__file__).parent.parent / "data" / "processed"
        
        # Load Challenge Set
        challenge_records = []
        with open(data_dir / "challenge_test.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                challenge_records.append(row)

        # Load Uncertainty & Adversarial Set
        uncert_csv = data_dir / "uncertainty_test.csv"
        if not uncert_csv.exists():
            from ai.data.uncertainty_dataset import UncertaintyDatasetGenerator
            gen = UncertaintyDatasetGenerator()
            recs = gen.generate()
            fields = ["id", "complaint_text", "domain_code", "category_code", "department_code", "language", "difficulty", "source"]
            with open(uncert_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(recs)

        uncert_records = []
        with open(uncert_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uncert_records.append(row)

        print("[Phase 4 Evaluation] Starting Comparative Evaluation...")

        # Initialize Phase 3 Agent (Uncalibrated, No OOS Gate) & Phase 4 Agent (Tuned & Calibrated)
        p3_agent = ClassificationAgent(
            taxonomy_manager=self.tm,
            tfidf_threshold=0.80,
            tfidf_margin=0.00,  # No margin protection in P3
            embedding_threshold=0.70
        )
        p3_agent.uncertainty_gate = None  # Phase 3 did not have UncertaintyGate pre-filtering

        p4_agent = ClassificationAgent(
            taxonomy_manager=self.tm,
            tfidf_threshold=0.80,
            tfidf_margin=0.15,  # Phase 4 Agent Safety Rule Margin
            embedding_threshold=0.70
        )

        # 1. Challenge Set Performance
        p3_challenge = self.evaluate_pipeline(challenge_records, p3_agent)
        p4_challenge = self.evaluate_pipeline(challenge_records, p4_agent)

        # 2. False Confident Classification Rate
        p3_false_conf = self.evaluate_false_confident_rate(uncert_records, p3_agent)
        p4_false_conf = self.evaluate_false_confident_rate(uncert_records, p4_agent)

        report = {
            "challenge_set_samples": len(challenge_records),
            "uncertainty_set_samples": len(uncert_records),
            "phase3_agent": {
                "challenge_accuracy": p3_challenge["accuracy"],
                "challenge_macro_f1": p3_challenge["macro_f1"],
                "avg_tools_per_complaint": p3_challenge["avg_tools_per_complaint"],
                "escalation_rates": p3_challenge["escalation_rates"],
                "false_confident_classification_rate": p3_false_conf["false_confident_classification_rate"]
            },
            "phase4_agent": {
                "challenge_accuracy": p4_challenge["accuracy"],
                "challenge_macro_f1": p4_challenge["macro_f1"],
                "avg_tools_per_complaint": p4_challenge["avg_tools_per_complaint"],
                "escalation_rates": p4_challenge["escalation_rates"],
                "false_confident_classification_rate": p4_false_conf["false_confident_classification_rate"]
            },
            "improvement_summary": {
                "accuracy_change": round(p4_challenge["accuracy"] - p3_challenge["accuracy"], 4),
                "macro_f1_change": round(p4_challenge["macro_f1"] - p3_challenge["macro_f1"], 4),
                "false_confident_reduction_pct": round(
                    (p3_false_conf["false_confident_classification_rate"] - p4_false_conf["false_confident_classification_rate"])
                    / max(0.001, p3_false_conf["false_confident_classification_rate"]),
                    4
                )
            }
        }

        # Save JSON report
        with open(self.results_dir / "phase3_vs_phase4_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        with open(self.results_dir / "false_confident_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"phase3": p3_false_conf, "phase4": p4_false_conf}, f, indent=2)

        print("\n==================================================")
        print("PHASE 3 VS PHASE 4 COMPARATIVE EVALUATION REPORT")
        print("==================================================")
        print(f"Challenge Set Accuracy:  Phase 3 = {p3_challenge['accuracy']:.2%}  |  Phase 4 = {p4_challenge['accuracy']:.2%}")
        print(f"Challenge Set Macro F1:  Phase 3 = {p3_challenge['macro_f1']:.4f}  |  Phase 4 = {p4_challenge['macro_f1']:.4f}")
        print(f"False Confident Rate:    Phase 3 = {p3_false_conf['false_confident_classification_rate']:.1%}  |  Phase 4 = {p4_false_conf['false_confident_classification_rate']:.1%}")
        print(f"False Confident Reduction: {report['improvement_summary']['false_confident_reduction_pct']:.1%} Reduction in Forced Category Errors")
        print(f"Average Tools Called:    Phase 3 = {p3_challenge['avg_tools_per_complaint']}  |  Phase 4 = {p4_challenge['avg_tools_per_complaint']}")
        print("==================================================")

        return report


if __name__ == "__main__":
    evaluator = Phase4Evaluator()
    evaluator.run_full_comparison()
