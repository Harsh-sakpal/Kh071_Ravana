"""
Phase 3 Evaluation & Comparison Engine for CivicFlow AI.

Evaluates and compares:
1. TF-IDF Baseline Tool Only
2. Multilingual Sentence Embedding Similarity Tool Only
3. Real Agentic ClassificationAgent (Dynamic Tool Escalation)

Evaluates on the unseen challenge dataset (ai/data/processed/challenge_test.csv)
and reports Accuracy, Macro F1, Tool Escalation %, Avg Tools per Complaint, and Latencies.
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


class Phase3Evaluator:
    """Evaluator comparing standalone tools against the Agentic Classification Pipeline."""

    def __init__(self):
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tm = TaxonomyManager()

    def evaluate_all(self, challenge_csv_path: str = None) -> Dict[str, Any]:
        """Runs evaluation comparing TF-IDF, Embedding, and Agentic Pipeline on challenge dataset."""
        if challenge_csv_path is None:
            challenge_csv_path = str(Path(__file__).parent.parent / "data" / "processed" / "challenge_test.csv")

        # First run challenge dataset generator if challenge_test.csv does not exist
        if not Path(challenge_csv_path).exists():
            from ai.data.challenge_dataset import ChallengeDatasetGenerator
            gen = ChallengeDatasetGenerator()
            ds = gen.generate_challenge_dataset()
            fields = ["id", "complaint_text", "domain_code", "category_code", "department_code", "language", "difficulty", "source"]
            with open(challenge_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(ds)

        test_records = []
        with open(challenge_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_records.append(row)

        print(f"[Phase 3 Evaluation] Loaded {len(test_records)} unseen challenge samples from {challenge_csv_path}...")

        # Initialize tools & agents
        tfidf_tool = TfidfClassificationTool()
        
        try:
            embedding_tool = EmbeddingSimilarityTool()
        except Exception:
            embedding_tool = None

        understanding_agent = UnderstandingAgent()
        classification_agent = ClassificationAgent(taxonomy_manager=self.tm)

        y_true = [r["category_code"] for r in test_records]

        # -------------------------------------------------------------
        # 1. MODE 1: Standalone TF-IDF Baseline Evaluation
        # -------------------------------------------------------------
        print("\n[Mode 1/3] Evaluating Standalone TF-IDF Tool on Challenge Set...")
        y_tfidf_pred = []
        tfidf_latencies = []

        for r in test_records:
            t0 = time.perf_counter()
            res = tfidf_tool.classify(r["complaint_text"])
            lat = (time.perf_counter() - t0) * 1000.0
            tfidf_latencies.append(lat)
            y_tfidf_pred.append(res.get("category", {}).get("code"))

        tfidf_acc = accuracy_score(y_true, y_tfidf_pred)
        t_p, t_r, t_f1, _ = precision_recall_fscore_support(y_true, y_tfidf_pred, average="macro", zero_division=0)
        t_wp, t_wr, t_wf1, _ = precision_recall_fscore_support(y_true, y_tfidf_pred, average="weighted", zero_division=0)

        tfidf_metrics = {
            "mode": "TFIDF_STANDALONE",
            "accuracy": round(float(tfidf_acc), 4),
            "macro_f1": round(float(t_f1), 4),
            "weighted_f1": round(float(t_wf1), 4),
            "avg_latency_ms": round(float(sum(tfidf_latencies) / len(tfidf_latencies)), 2),
            "avg_tools_per_complaint": 1.0,
        }

        # -------------------------------------------------------------
        # 2. MODE 2: Standalone Embedding Similarity Tool Evaluation
        # -------------------------------------------------------------
        embedding_metrics = {}
        y_emb_pred = []
        emb_latencies = []

        if embedding_tool and embedding_tool.is_ready:
            print("\n[Mode 2/3] Evaluating Standalone Multilingual Embedding Tool on Challenge Set...")
            for r in test_records:
                t0 = time.perf_counter()
                res = embedding_tool.classify(r["complaint_text"])
                lat = (time.perf_counter() - t0) * 1000.0
                emb_latencies.append(lat)
                y_emb_pred.append(res.get("category", {}).get("code"))

            emb_acc = accuracy_score(y_true, y_emb_pred)
            e_p, e_r, e_f1, _ = precision_recall_fscore_support(y_true, y_emb_pred, average="macro", zero_division=0)
            e_wp, e_wr, e_wf1, _ = precision_recall_fscore_support(y_true, y_emb_pred, average="weighted", zero_division=0)

            embedding_metrics = {
                "mode": "EMBEDDING_STANDALONE",
                "accuracy": round(float(emb_acc), 4),
                "macro_f1": round(float(e_f1), 4),
                "weighted_f1": round(float(e_wf1), 4),
                "avg_latency_ms": round(float(sum(emb_latencies) / len(emb_latencies)), 2),
                "avg_tools_per_complaint": 1.0,
            }
        else:
            print("\n[Mode 2/3] Embedding tool offline or loading.")
            embedding_metrics = {"mode": "EMBEDDING_STANDALONE", "status": "OFFLINE"}

        # -------------------------------------------------------------
        # 3. MODE 3: Agentic Classification Pipeline Evaluation
        # -------------------------------------------------------------
        print("\n[Mode 3/3] Evaluating Agentic Classification Agent (Confidence Escalation) on Challenge Set...")
        y_agent_pred = []
        agent_latencies = []
        tool_counts = []
        methods_used = Counter()

        for r in test_records:
            t0 = time.perf_counter()
            understanding = understanding_agent.understand(r["complaint_text"])
            res = classification_agent.classify(understanding, debug=True)
            lat = (time.perf_counter() - t0) * 1000.0

            agent_latencies.append(lat)
            cat_code = res.semantic_classification.category_code
            y_agent_pred.append(cat_code)
            methods_used[res.method] += 1

            # Count tools executed from trace
            trace_items = res.secondary_signals[0].get("trace", []) if res.secondary_signals else []
            tool_counts.append(max(1, len(trace_items)))

        agent_acc = accuracy_score(y_true, y_agent_pred)
        a_p, a_r, a_f1, _ = precision_recall_fscore_support(y_true, y_agent_pred, average="macro", zero_division=0)
        a_wp, a_wr, a_wf1, _ = precision_recall_fscore_support(y_true, y_agent_pred, average="weighted", zero_division=0)

        n_total = max(1, len(test_records))
        agent_metrics = {
            "mode": "AGENTIC_PIPELINE",
            "accuracy": round(float(agent_acc), 4),
            "macro_f1": round(float(a_f1), 4),
            "weighted_f1": round(float(a_wf1), 4),
            "avg_latency_ms": round(float(sum(agent_latencies) / len(agent_latencies)), 2),
            "avg_tools_per_complaint": round(float(sum(tool_counts) / len(tool_counts)), 2),
            "escalation_rates": {
                "tfidf_accepted_pct": round(methods_used.get("TFIDF", 0) / n_total, 4),
                "embedding_accepted_pct": round(methods_used.get("EMBEDDING", 0) / n_total, 4),
                "llm_fallback_accepted_pct": round(methods_used.get("LLM_FALLBACK", 0) / n_total, 4),
            }
        }

        # Save machine-readable JSON metrics
        with open(self.results_dir / "tfidf_metrics.json", "w", encoding="utf-8") as f:
            json.dump(tfidf_metrics, f, indent=2)

        with open(self.results_dir / "embedding_metrics.json", "w", encoding="utf-8") as f:
            json.dump(embedding_metrics, f, indent=2)

        with open(self.results_dir / "agent_metrics.json", "w", encoding="utf-8") as f:
            json.dump(agent_metrics, f, indent=2)

        comparison_summary = {
            "test_samples_count": len(test_records),
            "tfidf_standalone": tfidf_metrics,
            "embedding_standalone": embedding_metrics,
            "agentic_pipeline": agent_metrics,
        }

        with open(self.results_dir / "challenge_set_results.json", "w", encoding="utf-8") as f:
            json.dump(comparison_summary, f, indent=2)

        print("\n==================================================")
        print("PHASE 3 EVALUATION & AGENTIC COMPARISON SUMMARY")
        print("==================================================")
        print(f"Dataset: {len(test_records)} Unseen Challenge Samples")
        print(f"1. Standalone TF-IDF:     Accuracy={tfidf_metrics['accuracy']:.2%} | Macro F1={tfidf_metrics['macro_f1']:.4f} | Avg Latency={tfidf_metrics['avg_latency_ms']}ms")
        if embedding_metrics.get("accuracy"):
            print(f"2. Standalone Embedding: Accuracy={embedding_metrics['accuracy']:.2%} | Macro F1={embedding_metrics['macro_f1']:.4f} | Avg Latency={embedding_metrics['avg_latency_ms']}ms")
        print(f"3. Agentic Pipeline:     Accuracy={agent_metrics['accuracy']:.2%} | Macro F1={agent_metrics['macro_f1']:.4f} | Avg Latency={agent_metrics['avg_latency_ms']}ms")
        print(f"   -> Avg Tools Called:  {agent_metrics['avg_tools_per_complaint']} tools/complaint")
        print(f"   -> TF-IDF Early Accept: {agent_metrics['escalation_rates']['tfidf_accepted_pct']:.1%}")
        print(f"   -> Embedding Accepted:  {agent_metrics['escalation_rates']['embedding_accepted_pct']:.1%}")
        print(f"   -> LLM Fallback Pct:   {agent_metrics['escalation_rates']['llm_fallback_accepted_pct']:.1%}")
        print("==================================================")

        return comparison_summary


if __name__ == "__main__":
    evaluator = Phase3Evaluator()
    evaluator.evaluate_all()
