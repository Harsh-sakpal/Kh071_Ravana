"""
Machine Learning Error & Class Imbalance Analysis.
Generates confusion matrices, per-class error rates, and identifies worst-performing categories.
"""

import csv
import json
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent
from ai.taxonomy.taxonomy_manager import TaxonomyManager

def analyze_ml_errors():
    data_path = Path(__file__).parent.parent / "data" / "real" / "real_validation.csv"
    results_dir = Path(__file__).parent / "results"
    
    understanding = UnderstandingAgent()
    classifier = ClassificationAgent()
    tm = TaxonomyManager()
    
    y_true = []
    y_pred = []
    
    print("Running ML Error Analysis on real validation set...")
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            for row in reader:
                txt = row["complaint_text"]
                y_true.append(row["category_code"])
                
                u_out = understanding.understand(txt)
                c_out = classifier.classify(u_out, debug=False)
                y_pred.append(c_out.semantic_classification.category_code)
                
    if not y_true:
        print("No validation data found.")
        return
        
    # Generate classification report as dict
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Identify worst categories (F1 < 0.80)
    worst_categories = []
    for label, metrics in report.items():
        if isinstance(metrics, dict) and label not in ["accuracy", "macro avg", "weighted avg"]:
            if metrics["f1-score"] < 0.80:
                worst_categories.append({
                    "category": label,
                    "f1_score": metrics["f1-score"],
                    "support": metrics["support"]
                })
                
    # Class Imbalance Analysis
    class_counts = Counter(y_true)
    total_samples = len(y_true)
    imbalance_stats = {}
    for cat, count in class_counts.items():
        imbalance_stats[cat] = {
            "count": count,
            "percentage": round(count / total_samples, 4)
        }
                
    analysis_results = {
        "classification_report_summary": {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"]
        },
        "worst_performing_categories": sorted(worst_categories, key=lambda x: x["f1_score"]),
        "class_imbalance_analysis": imbalance_stats
    }
    
    with open(results_dir / "ml_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)
        
    print("ML Error Analysis Complete. Results saved to ml_error_analysis.json")

if __name__ == "__main__":
    analyze_ml_errors()
