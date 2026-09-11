"""
Model Generalization & Distribution Shift Analysis.
Compares vocabulary and category distributions between train, val, and challenge sets.
"""

import csv
import json
from pathlib import Path
from collections import Counter

def analyze_distribution_shift():
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent / "results"
    
    train_path = data_dir / "real" / "real_train.csv"
    val_path = data_dir / "real" / "real_validation.csv"
    challenge_path = data_dir / "processed" / "challenge_test.csv"
    
    def get_stats(path: Path):
        categories = []
        words = set()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cat = row.get("category_code")
                    if cat:
                        categories.append(cat)
                    txt = row.get("complaint_text", "")
                    for w in txt.lower().split():
                        words.add(w)
        return categories, words
        
    train_cats, train_vocab = get_stats(train_path)
    val_cats, val_vocab = get_stats(val_path)
    chal_cats, chal_vocab = get_stats(challenge_path)
    
    val_vocab_overlap = len(val_vocab.intersection(train_vocab)) / max(1, len(val_vocab))
    chal_vocab_overlap = len(chal_vocab.intersection(train_vocab)) / max(1, len(chal_vocab))
    
    report = {
        "train_set": {
            "samples": len(train_cats),
            "vocab_size": len(train_vocab)
        },
        "validation_set": {
            "samples": len(val_cats),
            "vocab_size": len(val_vocab),
            "vocab_overlap_with_train": round(val_vocab_overlap, 4)
        },
        "challenge_set": {
            "samples": len(chal_cats),
            "vocab_size": len(chal_vocab),
            "vocab_overlap_with_train": round(chal_vocab_overlap, 4)
        }
    }
    
    with open(results_dir / "distribution_shift_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("Distribution Shift Analysis Complete.")

if __name__ == "__main__":
    analyze_distribution_shift()
