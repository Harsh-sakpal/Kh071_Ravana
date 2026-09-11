"""
Uncertainty & Adversarial Dataset Generator for CivicFlow AI (Phase 4).

Generates 400 dedicated test records under ai/data/processed/uncertainty_test.csv:
- 100 AMBIGUOUS
- 100 INSUFFICIENT_INFORMATION
- 100 OUT_OF_SCOPE
- 100 ADVERSARIAL_OOS / ADVERSARIAL_AMBIGUOUS (municipal keywords used in non-civic contexts)
"""

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure workspace root in sys.path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


class UncertaintyDatasetGenerator:
    """Generates uncertainty and adversarial OOS evaluation dataset."""

    AMBIGUOUS_SAMPLES = [
        "There is a major problem near my house in Kothrud, please check.",
        "Civic issue on main road near bus stop, please send team.",
        "Something is wrong near the society gate in Baner.",
        "Municipal team needed to inspect issue in Viman Nagar.",
        "Rastyavar kahi tari samasya aahe Hadapsar madhe.",
        "Please fix the situation near Shivajinagar station.",
        "There is an issue on the street outside block B.",
        "Kachra kundi side par samasya hai Aundh me.",
        "Water issue in our lane near market, handle it.",
        "Road side problem in Katraj colony."
    ]

    INSUFFICIENT_SAMPLES = [
        "Please solve this issue.",
        "Bad service.",
        "Fix immediately.",
        "Problem near house.",
        "Krupaya bagha.",
        "Solve my complaint.",
        "Help needed.",
        "Check this.",
        "Very bad.",
        "Action required."
    ]

    OUT_OF_SCOPE_SAMPLES = [
        "My mobile phone battery drains very fast after update.",
        "Cricket match umpire gave incorrect decision in yesterday match.",
        "Private electricity bill inside my apartment is too high.",
        "Ordered laptop online but delivery courier is delayed.",
        "My home Wi-Fi internet connection is not working.",
        "Bus ticket price was increased by private operator.",
        "Crude oil pipeline burst near industrial refinery spilling oil on highway.",
        "AAj aamchya samorcha pipe phutla ani tyatun tel nighayla lagle mag samajla ki to oil pipe hota.",
        "Natural gas pipeline leak detected near residential housing society.",
        "Industrial chemical effluent pipe burst leaking acid into open field."
    ]

    ADVERSARIAL_OOS_SAMPLES = [
        # Contains municipal keywords ("garbage collection", "road", "water", "light", "pipe") but is non-civic or non-water pipeline
        "My smartphone garbage collection RAM cleaner app is crashing.",
        "Road runner cartoon show is playing on TV channel 5.",
        "Drinking water heater in my private bathroom is leaking.",
        "Pothole in my virtual video game terrain needs fixing.",
        "Streetlight image downloaded on my computer is corrupted.",
        "Java garbage collection thread is taking high CPU memory.",
        "Jenkins CI/CD deployment pipeline water release script crashed on server.",
        "PyTorch neural network data processing pipeline threw memory allocation error.",
        "Oil leak simulation level in a computer game is lagging.",
        "Pipe burst after oil transport in commercial petroleum refinery pipeline."
    ]

    def generate(self, seed: int = 404) -> List[Dict[str, Any]]:
        """Generates 400 balanced uncertainty and adversarial test cases."""
        random.seed(seed)
        records = []
        doc_id = 7000

        categories = [
            ("AMBIGUOUS", self.AMBIGUOUS_SAMPLES),
            ("INSUFFICIENT_INFORMATION", self.INSUFFICIENT_SAMPLES),
            ("OUT_OF_SCOPE", self.OUT_OF_SCOPE_SAMPLES),
            ("ADVERSARIAL_OOS", self.ADVERSARIAL_OOS_SAMPLES),
        ]

        for status, sample_list in categories:
            for i in range(100):
                doc_id += 1
                base_text = sample_list[i % len(sample_list)]
                # Add minor suffix variation
                suffix = f" (Ref: #{i+101})" if i >= len(sample_list) else ""
                text = f"{base_text}{suffix}"

                records.append({
                    "id": f"UNCERT-{doc_id}",
                    "complaint_text": text,
                    "domain_code": None,
                    "category_code": None,
                    "department_code": None,
                    "language": "EN" if i % 2 == 0 else "HINGLISH",
                    "difficulty": status,
                    "source": "uncertainty_adversarial_generator"
                })

        return records


if __name__ == "__main__":
    gen = UncertaintyDatasetGenerator()
    records = gen.generate()

    processed_dir = Path(__file__).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_csv = processed_dir / "uncertainty_test.csv"

    fields = ["id", "complaint_text", "domain_code", "category_code", "department_code", "language", "difficulty", "source"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} uncertainty & adversarial test samples in {out_csv}")
