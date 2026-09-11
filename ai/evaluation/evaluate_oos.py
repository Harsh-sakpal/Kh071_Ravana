"""
Evaluates Out-Of-Scope Safety Policy on out_of_scope_attack_test.csv.
"""

import csv
import json
from pathlib import Path
from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_oos():
    data_path = Path(__file__).parent.parent / "data" / "processed" / "out_of_scope_attack_test.csv"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    understanding = UnderstandingAgent()
    classifier = ClassificationAgent()

    y_true_oos = []
    y_pred_oos = []
    false_positives = []
    false_negatives = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = row["complaint_text"]
            expected = row["expected_category"]
            
            # Treat OUT_OF_SCOPE and AMBIGUOUS as True for OOS/Safety detection, else False (Valid Municipal)
            is_true_oos = (expected in ["OUT_OF_SCOPE", "AMBIGUOUS"])
            y_true_oos.append(is_true_oos)

            # Predict
            u_out = understanding.understand(txt)
            c_out = classifier.classify(u_out, debug=False)
            
            is_pred_oos = (c_out.status in ["OUT_OF_SCOPE", "AMBIGUOUS"])
            y_pred_oos.append(is_pred_oos)

            if is_true_oos and not is_pred_oos:
                false_negatives.append({
                    "text": txt,
                    "expected": expected,
                    "predicted_status": c_out.status,
                    "predicted_category": c_out.semantic_classification.category_code
                })
            elif not is_true_oos and is_pred_oos:
                false_positives.append({
                    "text": txt,
                    "expected": expected,
                    "predicted_status": c_out.status
                })

    precision = precision_score(y_true_oos, y_pred_oos, zero_division=0)
    recall = recall_score(y_true_oos, y_pred_oos, zero_division=0)
    f1 = f1_score(y_true_oos, y_pred_oos, zero_division=0)

    report = {
        "oos_precision": round(float(precision), 4),
        "oos_recall": round(float(recall), 4),
        "oos_f1": round(float(f1), 4),
        "false_municipal_classification_rate": round(len(false_negatives) / max(1, len(y_true_oos)), 4),
        "false_oos_classification_rate": round(len(false_positives) / max(1, len([y for y in y_true_oos if not y])), 4),
        "false_positives_examples": false_positives,
        "false_negatives_examples": false_negatives
    }

    with open(results_dir / "oos_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n==================================================")
    print("OOS SAFETY POLICY EVALUATION")
    print("==================================================")
    print(f"OOS Precision: {precision:.2%}")
    print(f"OOS Recall:    {recall:.2%}")
    print(f"OOS F1 Score:  {f1:.4f}")
    print("==================================================\n")

if __name__ == "__main__":
    evaluate_oos()
