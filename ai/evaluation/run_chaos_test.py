"""
End-To-End Chaos Test Suite.
Streams a mixed random barrage of valid, OOS, adversarial, and malformed inputs 
through the pipeline to ensure 0 unhandled crashes and 0 taxonomy violations.
"""

import csv
import json
import random
from pathlib import Path

from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent
from ai.taxonomy.taxonomy_manager import TaxonomyManager

def run_chaos_test():
    understanding = UnderstandingAgent()
    classifier = ClassificationAgent()
    tm = TaxonomyManager()
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load Real Val Data
    val_path = data_dir / "real" / "real_validation.csv"
    val_data = []
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val_data.append(row["complaint_text"])
                
    # Load OOS Data
    oos_path = data_dir / "processed" / "out_of_scope_attack_test.csv"
    oos_data = []
    if oos_path.exists():
        with open(oos_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                oos_data.append(row["complaint_text"])

    # Synthetic Malformed/Adversarial
    chaos_pool = val_data[:200] + oos_data + [
        "garbage garbage garbage",
        "123 456 7890",
        "",
        None,
        "   \n",
        "Marathi text: रस्ता खराब आहे",
        "Roman Marathi: rasta kharab aahe",
        "Hinglish: mera road kharab hai please fix",
        "Multiple: Road has potholes and water is leaking everywhere",
        "Adversarial: P0th0le in r0ad",
        "Very long: " + ("pothole " * 1000)
    ]
    
    # Shuffle for chaos
    random.seed(42)
    random.shuffle(chaos_pool)
    
    crashes = 0
    taxonomy_violations = 0
    safe_completions = 0
    
    print("==================================================")
    print("STARTING CHAOS TEST STREAM")
    print("==================================================")
    
    all_categories = set(tm.get_categories().keys())
    all_domains = set(tm.get_domains())
    
    import time
    start = time.time()
    
    for i, txt in enumerate(chaos_pool):
        try:
            u_out = understanding.understand(txt if txt is not None else "")
            c_out = classifier.classify(u_out, debug=False)
            
            # Check Taxonomy Violation
            stat = c_out.status
            dom = c_out.semantic_classification.domain_code
            cat = c_out.semantic_classification.category_code
            
            if stat == "CLASSIFIED":
                if cat not in all_categories or dom not in all_domains:
                    taxonomy_violations += 1
                    print(f"[VIOLATION] Invalid taxonomy output: {dom} -> {cat}")
                else:
                    safe_completions += 1
            else:
                # OOS or AMBIGUOUS
                safe_completions += 1
                
        except Exception as e:
            crashes += 1
            print(f"[CRASH] {e} on input: {txt[:50]}")
            
    latency = time.time() - start
            
    print("==================================================")
    print("CHAOS TEST COMPLETE")
    print("==================================================")
    print(f"Total Inputs:          {len(chaos_pool)}")
    print(f"Safe Completions:      {safe_completions}")
    print(f"Crashes:               {crashes}")
    print(f"Taxonomy Violations:   {taxonomy_violations}")
    print(f"Avg Latency per Input: {latency/len(chaos_pool):.4f}s")
    print("==================================================\n")
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "chaos_test_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_inputs": len(chaos_pool),
            "safe_completions": safe_completions,
            "crashes": crashes,
            "taxonomy_violations": taxonomy_violations,
            "avg_latency": round(latency/len(chaos_pool), 4)
        }, f, indent=2)

if __name__ == "__main__":
    run_chaos_test()
