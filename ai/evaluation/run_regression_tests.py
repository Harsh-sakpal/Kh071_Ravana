"""
Final Regression Testing Suite.
Compares performance on real validation data (before/after logic hardening).
"""

import csv
import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent

def run_regression_tests():
    data_path = Path(__file__).parent.parent / "data" / "real" / "real_validation.csv"
    results_dir = Path(__file__).parent / "results"
    
    understanding = UnderstandingAgent()
    classifier = ClassificationAgent()
    
    y_true_dom = []
    y_true_cat = []
    y_pred_dom = []
    y_pred_cat = []
    
    print("==================================================")
    print("STARTING REGRESSION TEST (real_validation.csv)")
    print("==================================================")
    
    start = time.time()
    
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            total = len(reader)
            
            for i, row in enumerate(reader):
                txt = row["complaint_text"]
                y_true_dom.append(row["domain_code"])
                y_true_cat.append(row["category_code"])
                
                u_out = understanding.understand(txt)
                c_out = classifier.classify(u_out, debug=False)
                
                y_pred_dom.append(c_out.semantic_classification.domain_code)
                y_pred_cat.append(c_out.semantic_classification.category_code)
                
    latency = time.time() - start
    
    dom_acc = accuracy_score(y_true_dom, y_pred_dom) if y_true_dom else 0.0
    cat_acc = accuracy_score(y_true_cat, y_pred_cat) if y_true_cat else 0.0
    macro_f1 = f1_score(y_true_cat, y_pred_cat, average="macro", zero_division=0) if y_true_cat else 0.0
    
    print(f"Domain Accuracy:   {dom_acc:.2%}")
    print(f"Category Accuracy: {cat_acc:.2%}")
    print(f"Category Macro F1: {macro_f1:.4f}")
    print(f"Avg Latency:       {latency/max(1, len(y_true_dom)):.4f}s")
    print("==================================================\n")
    
    with open(results_dir / "regression_test_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "samples": len(y_true_dom),
            "domain_accuracy": round(float(dom_acc), 4),
            "category_accuracy": round(float(cat_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "avg_latency": round(latency/max(1, len(y_true_dom)), 4)
        }, f, indent=2)

if __name__ == "__main__":
    run_regression_tests()
