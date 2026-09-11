"""
Inference Demonstration Script for CivicFlow AI TF-IDF Baseline.
Runs inference on representative complaints and hard-negative examples, demonstrating contract output.
"""

import json
import sys
from pathlib import Path

# Ensure root workspace in sys.path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.tools.tfidf_tool import TfidfClassificationTool

REPRESENTATIVE_TEST_CASES = [
    # 1. Solid Waste (EN)
    "Garbage vehicle has not arrived for three days in society. Trash is overflowing near main gate.",
    # 2. Solid Waste (Marathi)
    "Amchya gallyat kachra kundi tutli aahe aani sagla kachra rastyavar aala aahe.",
    # 3. Water Leakage (EN)
    "Main drinking water pipeline burst near bus stop. Thousands of liters of water spilling continuously.",
    # 4. Roads Pothole (EN)
    "Huge dangerous pothole near school signal causing two wheelers to slip daily.",
    # 5. Street Light Outage (Hinglish)
    "Rastyavarchi street light maagchya aathavdya pasun chalat nahiye, full andhar aahe.",
    # 6. Electrical Hazard (EN)
    "Live electrical wire hanging low near children park gate posing severe hazard.",
    # 7. Drainage Manhole (Hinglish)
    "Open gatar manhole cover missing near bus stop. Severe risk of accident.",
    # 8. Property Tax Unmapped (EN)
    "Property tax online bill double calculation error. Received incorrect penalty notice.",
    # 9. Trade License Violation Unmapped (EN)
    "Commercial workshop operating illegally without trade license in residential area.",
    # 10. Fire Safety Hazard Cross Agency (EN)
    "Commercial building emergency staircase blocked with heavy wooden boxes."
]

HARD_NEGATIVE_TEST_CASES = [
    # Pair 1: Garbage Collection vs Illegal Dumping
    ("SAN-GARB", "Garbage van did not visit our building this morning for collection."),
    ("SAN-DUMP", "Unidentified trucks dumping commercial waste in open plot near market."),
    # Pair 2: Water Leakage vs Water Contamination vs Sewer Overflow
    ("WTR-LEAK", "Drinking water pipeline leaking heavily under the main road."),
    ("WTR-QUAL", "Dirty black tap water coming from tap smells like sewage."),
    ("SEW-BLOCK", "Sewage line clogged and filthy gutter water overflowing onto street."),
    # Pair 3: Pothole vs Broken Footpath vs Dug Up Road
    ("RDW-POT", "Deep pothole crater on road damaging car tyres."),
    ("RDW-WALK", "Footpath paver blocks broken and missing causing elderly to trip."),
    ("RDW-RD", "Road dug up for pipeline was abandoned unpaved without asphalt layer.")
]


def run_demo():
    tool = TfidfClassificationTool()
    print("==================================================")
    print("CIVICFLOW AI - TF-IDF INFERENCE DEMONSTRATION")
    print("==================================================")

    print("\n--- 1. REPRESENTATIVE COMPLAINT INFERENCE ---")
    for idx, text in enumerate(REPRESENTATIVE_TEST_CASES, 1):
        res = tool.classify(text)
        dom = res["domain"]["code"]
        cat = res["category"]["code"]
        conf = res["overall_confidence"]
        map_type = res["administrative_mapping"]["type"]
        dept = res["administrative_mapping"]["department_code"]

        print(f"\n[Case {idx}] Text: \"{text}\"")
        print(f"  -> Predicted Domain:   {dom} ({res['domain']['confidence']:.1%})")
        print(f"  -> Predicted Category: {cat} ({res['category']['confidence']:.1%})")
        print(f"  -> Overall Confidence: {conf:.1%}")
        print(f"  -> Admin Mapping:      {map_type} (Dept: {dept})")

    print("\n--- 2. HARD-NEGATIVE PAIRS INFERENCE ---")
    for expected_cat, text in HARD_NEGATIVE_TEST_CASES:
        res = tool.classify(text)
        pred_cat = res["category"]["code"]
        is_correct = "PASS" if pred_cat == expected_cat else "FAIL"
        print(f"Expected: {expected_cat:10s} | Predicted: {pred_cat:10s} | Status: [{is_correct}] | Text: \"{text[:65]}...\"")

    print("==================================================")


if __name__ == "__main__":
    run_demo()
