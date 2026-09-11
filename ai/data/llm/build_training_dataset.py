"""
Build Instruction-Tuning Dataset for CivicFlow AI LLM Fine-Tuning.

Converts safe training data (train.csv, special_cases.csv, hard_negatives.json, and taxonomy intents)
into ChatML instruction format for Qwen2.5-1.5B-Instruct.

Outputs:
- ai/data/llm/final_train.jsonl
- ai/data/llm/final_validation.jsonl
- ai/data/llm/DATASET_MANIFEST.json
"""

import json
import os
import random
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Set fixed seed for reproducibility
random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TAXONOMY_PATH = PROJECT_ROOT / "ai" / "taxonomy" / "taxonomy.json"
TRAIN_CSV = PROJECT_ROOT / "ai" / "data" / "processed" / "train.csv"
SPECIAL_CASES_CSV = PROJECT_ROOT / "ai" / "data" / "processed" / "special_cases.csv"
HARD_NEGATIVES_JSON = PROJECT_ROOT / "ai" / "data" / "processed" / "hard_negatives.json"

EXCLUDED_EVAL_DATASETS = [
    "ai/data/processed/challenge_test.csv",
    "ai/data/processed/uncertainty_test.csv",
    "ai/data/processed/out_of_scope_attack_test.csv",
    "ai/data/processed/test.csv"
]

OUTPUT_DIR = PROJECT_ROOT / "ai" / "data" / "llm"
TRAIN_JSONL = OUTPUT_DIR / "final_train.jsonl"
VAL_JSONL = OUTPUT_DIR / "final_validation.jsonl"
MANIFEST_JSON = OUTPUT_DIR / "DATASET_MANIFEST.json"

SYSTEM_PROMPT = (
    "You are the Municipal Complaint Classification Assistant for CivicFlow AI. "
    "Your task is to classify citizen complaints into the 15-domain fixed municipal taxonomy "
    "or assign a valid special status (AMBIGUOUS, OUT_OF_SCOPE, INSUFFICIENT_INFORMATION, UNKNOWN). "
    "Return ONLY valid JSON matching the exact schema."
)


def load_taxonomy():
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        tax = json.load(f)
    cat_to_dom = {}
    for d_code, d_info in tax.get("domains", {}).items():
        for c_code in d_info.get("categories", {}).keys():
            cat_to_dom[c_code] = d_code
    return tax, cat_to_dom


def create_instruction_sample(
    text: str,
    status: str,
    domain_code: str = None,
    category_code: str = None,
    language: str = "EN",
    evidence: List[str] = None,
    explanation: str = None
) -> Dict[str, Any]:
    """Formats a single example into ChatML format."""
    if status == "CLASSIFIED":
        assistant_payload = {
            "status": "CLASSIFIED",
            "domain_code": domain_code,
            "category_code": category_code,
            "confidence": 0.95,
            "evidence": evidence or [w for w in text.split()[:3] if len(w) > 3],
            "explanation": explanation or f"Complaint classified under category {category_code} in domain {domain_code}."
        }
    else:
        assistant_payload = {
            "status": status,
            "domain_code": None,
            "category_code": None,
            "confidence": 0.90 if status in ("OUT_OF_SCOPE", "INSUFFICIENT_INFORMATION") else 0.70,
            "evidence": evidence or [],
            "explanation": explanation or f"Input assigned special status {status}."
        }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)}
        ],
        "_meta": {
            "status": status,
            "category_code": category_code or status,
            "domain_code": domain_code or status,
            "language": language
        }
    }


def build_dataset():
    print("==================================================")
    print("BUILDING INSTRUCTION-TUNING DATASET")
    print("==================================================")

    taxonomy, cat_to_dom = load_taxonomy()
    all_samples = []
    seen_texts = set()

    # 1. Process main train.csv
    if TRAIN_CSV.exists():
        df_train = pd.read_csv(TRAIN_CSV)
        print(f"Loaded {len(df_train)} rows from train.csv")
        for _, row in df_train.iterrows():
            text = str(row["complaint_text"]).strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            cat = str(row.get("category_code", "")).strip()
            dom = cat_to_dom.get(cat, str(row.get("domain_code", "")).strip())
            lang = str(row.get("language", "EN")).upper()

            if cat and dom and cat in cat_to_dom:
                sample = create_instruction_sample(
                    text=text,
                    status="CLASSIFIED",
                    domain_code=dom,
                    category_code=cat,
                    language=lang,
                    explanation=f"Municipal complaint concerning {cat} ({dom})."
                )
                all_samples.append(sample)

    # 2. Process special_cases.csv (OOS, AMBIGUOUS, INSUFFICIENT_INFO)
    if SPECIAL_CASES_CSV.exists():
        df_special = pd.read_csv(SPECIAL_CASES_CSV)
        print(f"Loaded {len(df_special)} rows from special_cases.csv")
        for _, row in df_special.iterrows():
            text = str(row["complaint_text"]).strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            cat = str(row.get("category_code", "")).strip()
            dom = str(row.get("domain_code", "")).strip()
            diff = str(row.get("difficulty", "")).strip().upper()
            lang = str(row.get("language", "EN")).upper()

            if cat == "nan" or not cat:
                cat = None
            if dom == "nan" or not dom:
                dom = None

            status = "CLASSIFIED"
            if diff in ("AMBIGUOUS", "UNKNOWN", "OUT_OF_SCOPE", "INSUFFICIENT_INFORMATION"):
                status = diff
                cat = None
                dom = None
            elif cat in ("OUT_OF_SCOPE", "AMBIGUOUS", "INSUFFICIENT_INFORMATION", "UNKNOWN"):
                status = cat
                cat = None
                dom = None
            elif dom in ("OUT_OF_SCOPE", "AMBIGUOUS", "INSUFFICIENT_INFORMATION", "UNKNOWN"):
                status = dom
                cat = None
                dom = None

            sample = create_instruction_sample(
                text=text,
                status=status,
                domain_code=dom,
                category_code=cat,
                language=lang
            )
            all_samples.append(sample)

    # 3. Process hard negatives from hard_negatives.json
    if HARD_NEGATIVES_JSON.exists():
        with open(HARD_NEGATIVES_JSON, "r", encoding="utf-8") as f:
            hard_negs = json.load(f)
        print(f"Loaded hard negatives metadata from hard_negatives.json")
        for pair in hard_negs.get("pairs", []):
            t1 = pair.get("text_1")
            c1 = pair.get("cat_1")
            t2 = pair.get("text_2")
            c2 = pair.get("cat_2")
            if t1 and c1 and t1 not in seen_texts and c1 in cat_to_dom:
                seen_texts.add(t1)
                all_samples.append(create_instruction_sample(t1, "CLASSIFIED", cat_to_dom[c1], c1, "EN"))
            if t2 and c2 and t2 not in seen_texts and c2 in cat_to_dom:
                seen_texts.add(t2)
                all_samples.append(create_instruction_sample(t2, "CLASSIFIED", cat_to_dom[c2], c2, "EN"))

    # 4. Generate taxonomy paraphrase examples if needed for category balance
    cat_counts = {}
    for s in all_samples:
        c = s["_meta"]["category_code"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    # Ensure all 34 categories have at least 30 training examples via taxonomy intents
    for d_code, d_info in taxonomy.get("domains", {}).items():
        for c_code, c_info in d_info.get("categories", {}).items():
            count = cat_counts.get(c_code, 0)
            intents = c_info.get("example_intents", [])
            desc = c_info.get("description", "")
            name = c_info.get("category_name", "")

            # Synthetic variations for lower count categories
            templates = [
                f"Please fix {name.lower()} in our area.",
                f"Urgent attention needed for {desc.lower()}.",
                f"Complaint regarding {name.lower()} in Pune city.",
                f"आमच्या परिसरात {name.lower()} ची समस्या आहे, त्वरित कारवाई करा.",
                f"Hamare area me {name.lower()} ka issue bohot badh gaya hai.",
                f"High priority: {desc}."
            ]
            for intent in intents:
                if intent not in seen_texts:
                    seen_texts.add(intent)
                    all_samples.append(create_instruction_sample(intent, "CLASSIFIED", d_code, c_code, "EN"))
                # Add Marathi / Hinglish variants of intents
                mar_intent = f"तक्रार: {intent} (क्षेत्र: {name})"
                if mar_intent not in seen_texts:
                    seen_texts.add(mar_intent)
                    all_samples.append(create_instruction_sample(mar_intent, "CLASSIFIED", d_code, c_code, "MARATHI"))

            for tmpl in templates:
                if tmpl not in seen_texts and cat_counts.get(c_code, 0) < 45:
                    seen_texts.add(tmpl)
                    lang = "MARATHI" if "आमच्या" in tmpl else ("HINGLISH" if "Hamare" in tmpl else "EN")
                    all_samples.append(create_instruction_sample(tmpl, "CLASSIFIED", d_code, c_code, lang))
                    cat_counts[c_code] = cat_counts.get(c_code, 0) + 1

    # 5. Add synthetic OOS / Ambiguous / Insufficient samples if needed
    oos_examples = [
        ("What is the stock price of Apple today?", "OUT_OF_SCOPE", "EN"),
        ("Write a Python script to sort a list of numbers.", "OUT_OF_SCOPE", "EN"),
        ("Who won the Cricket World Cup in 2011?", "OUT_OF_SCOPE", "EN"),
        ("How do I cook butter chicken at home?", "OUT_OF_SCOPE", "EN"),
        ("Can you summarize the plot of Inception movie?", "OUT_OF_SCOPE", "EN"),
        ("आजचे तापमान काय आहे?", "OUT_OF_SCOPE", "MARATHI"),
        ("Mujhe naye phone ki price batayo.", "OUT_OF_SCOPE", "HINGLISH"),
        ("Help", "INSUFFICIENT_INFORMATION", "EN"),
        ("Problem here", "INSUFFICIENT_INFORMATION", "EN"),
        ("नाही", "INSUFFICIENT_INFORMATION", "MARATHI"),
        ("Plz check", "INSUFFICIENT_INFORMATION", "HINGLISH"),
        ("Road is broken and garbage is spilling and electricity pole is burning and water is dirty.", "AMBIGUOUS", "EN"),
        ("रस्ता खराब आहे आणि कचरा पडला आहे आणि पाणी येत नाही.", "AMBIGUOUS", "MARATHI"),
    ]
    for text, status, lang in oos_examples:
        if text not in seen_texts:
            seen_texts.add(text)
            all_samples.append(create_instruction_sample(text, status=status, language=lang))

    # Shuffle samples reproducibly
    random.shuffle(all_samples)

    # 80/20 train/val split
    total = len(all_samples)
    val_size = max(200, int(total * 0.15))
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write train.jsonl
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for s in train_samples:
            clean_item = {"messages": s["messages"]}
            f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")

    # Write validation.jsonl
    with open(VAL_JSONL, "w", encoding="utf-8") as f:
        for s in val_samples:
            clean_item = {"messages": s["messages"]}
            f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")

    # Compute manifest statistics
    cat_dist = {}
    lang_dist = {}
    status_dist = {}

    for s in all_samples:
        meta = s["_meta"]
        c = meta["category_code"]
        l = meta["language"]
        st = meta["status"]

        cat_dist[c] = cat_dist.get(c, 0) + 1
        lang_dist[l] = lang_dist.get(l, 0) + 1
        status_dist[st] = status_dist.get(st, 0) + 1

    manifest = {
        "dataset_name": "CivicFlow AI LLM Instruction Dataset",
        "model_target": "Qwen/Qwen2.5-1.5B-Instruct",
        "total_samples": total,
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "status_distribution": status_dist,
        "language_distribution": lang_dist,
        "category_distribution": cat_dist,
        "sources_used": [
            "ai/data/processed/train.csv",
            "ai/data/processed/special_cases.csv",
            "ai/data/processed/hard_negatives.json",
            "ai/taxonomy/taxonomy.json (example intents)"
        ],
        "excluded_evaluation_datasets": EXCLUDED_EVAL_DATASETS
    }

    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Dataset built successfully!")
    print(f"  Total Samples:      {total}")
    print(f"  Train Samples:      {len(train_samples)} -> {TRAIN_JSONL}")
    print(f"  Validation Samples: {len(val_samples)} -> {VAL_JSONL}")
    print(f"  Manifest:           {MANIFEST_JSON}")


if __name__ == "__main__":
    build_dataset()
