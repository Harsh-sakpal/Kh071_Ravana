"""
Manual Test Mode CLI for CivicFlow AI — BASE vs FINE-TUNED Model Comparison.

Includes external anti-hallucination guardrails:
1. Strict taxonomy validation (rejects non-existent codes)
2. Confidence threshold gating (< 0.60 → AMBIGUOUS)
3. Evidence grounding filter (removes fabricated evidence terms)

Appends test logs to ai/evaluation/manual_llm_tests.jsonl.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = PROJECT_ROOT / "ai" / "taxonomy" / "taxonomy.json"
MANUAL_TEST_LOG = PROJECT_ROOT / "ai" / "evaluation" / "manual_llm_tests.jsonl"


class TaxonomyValidator:
    def __init__(self):
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
            self.tax = json.load(f)
        self.domains = set(self.tax.get("domains", {}).keys())
        self.categories = {}
        for d_code, d_info in self.tax.get("domains", {}).items():
            for c_code in d_info.get("categories", {}).keys():
                self.categories[c_code] = d_code

    def is_valid(self, status: str, domain_code: str, category_code: str) -> bool:
        if status in ("AMBIGUOUS", "UNKNOWN", "OUT_OF_SCOPE", "INSUFFICIENT_INFORMATION"):
            return True
        if status == "CLASSIFIED":
            if category_code in self.categories and domain_code in self.domains:
                return self.categories[category_code] == domain_code
        return False


# ============================================================
# ANTI-HALLUCINATION GUARDRAILS (applied after every inference)
# ============================================================

def apply_guardrails(result: Dict[str, Any], original_text: str, validator: TaxonomyValidator) -> Dict[str, Any]:
    """
    External post-inference guardrails to catch and fix hallucinations.

    Guardrail 1: Taxonomy Code Validation
        - If status=CLASSIFIED but domain/category are not in taxonomy.json → degrade to AMBIGUOUS
    Guardrail 2: Confidence Threshold
        - If status=CLASSIFIED but confidence < 0.60 → degrade to AMBIGUOUS
    Guardrail 3: Evidence Grounding
        - Remove any evidence term that does NOT appear in the user's original text
    """
    status = result.get("status", "UNKNOWN")
    domain_code = result.get("domain_code")
    category_code = result.get("category_code")
    confidence = result.get("confidence", 0.0)
    evidence = result.get("evidence", [])
    explanation = result.get("explanation", "")
    guardrail_flags = []

    # --- Guardrail 1: Taxonomy Code Validation ---
    if status == "CLASSIFIED":
        if not validator.is_valid(status, domain_code, category_code):
            guardrail_flags.append(
                f"BLOCKED: Hallucinated code {domain_code}/{category_code} not in taxonomy → degraded to AMBIGUOUS"
            )
            status = "AMBIGUOUS"
            domain_code = None
            category_code = None
            confidence = round(confidence * 0.5, 4)

    # --- Guardrail 2: Confidence Threshold ---
    if status == "CLASSIFIED" and confidence < 0.60:
        guardrail_flags.append(
            f"BLOCKED: Low confidence {confidence} < 0.60 → degraded to AMBIGUOUS"
        )
        status = "AMBIGUOUS"
        domain_code = None
        category_code = None

    # --- Guardrail 3: Evidence Grounding ---
    text_lower = original_text.lower()
    verified_evidence = []
    hallucinated_evidence = []
    for item in evidence:
        item_str = str(item).strip()
        if not item_str:
            continue
        # Check if any word (len > 2) from this evidence term is present in original text
        words = [w for w in item_str.lower().split() if len(w) > 2]
        if words and any(w in text_lower for w in words):
            verified_evidence.append(item_str)
        else:
            hallucinated_evidence.append(item_str)

    if hallucinated_evidence:
        guardrail_flags.append(
            f"STRIPPED: Fabricated evidence removed: {hallucinated_evidence}"
        )

    # Clear domain/category for non-CLASSIFIED statuses
    if status != "CLASSIFIED":
        domain_code = None
        category_code = None

    return {
        "status": status,
        "domain_code": domain_code,
        "category_code": category_code,
        "confidence": confidence,
        "evidence": verified_evidence,
        "explanation": explanation,
        "guardrail_flags": guardrail_flags,
    }


# ============================================================
# MOCK MODEL INFERENCE FUNCTIONS
# ============================================================

# Roman Marathi / Hinglish keyword mappings for better multilingual coverage
WATER_KEYWORDS = ["water", "pani", "panaycha", "pump", "jal", "nal", "nala", "pipeline", "nalavahini",
                   "पाणी", "पंप", "जलवाहिनी", "नळ", "leak", "futla"]
GARBAGE_KEYWORDS = ["garbage", "kachra", "gadi", "dustbin", "dump", "कचरा", "कचराकुंडी", "गाडी", "साफसफाई"]
ROAD_KEYWORDS = ["pothole", "road", "rasta", "khadda", "footpath", "खड्डा", "रस्ता", "sadak"]
LIGHT_KEYWORDS = ["light", "pole", "lamp", "streetlight", "diva", "vij", "दिवा", "विजेचा", "खांब"]
SEWER_KEYWORDS = ["sewer", "drain", "gutter", "nali", "nala", "sewage", "गटार", "नाली", "मॅनहोल"]
TREE_KEYWORDS = ["tree", "park", "garden", "jhad", "झाड", "बाग", "उद्यान"]
OOS_KEYWORDS = ["weather", "python", "movie", "cricket", "song", "recipe", "game", "stock", "phone",
                "school", "exam", "birthday", "shopping", "amazon"]


def match_keywords(text_lower: str, keywords: List[str]) -> List[str]:
    """Returns matched keywords found in text."""
    return [kw for kw in keywords if kw in text_lower]


def mock_base_inference(text: str, validator: TaxonomyValidator) -> Dict[str, Any]:
    """Simulates BASE Qwen2.5-1.5B zero-shot inference (no task-specific fine-tuning)."""
    text_lower = text.lower()

    # Base model only recognizes English keywords reliably
    if match_keywords(text_lower, ["pothole", "road"]):
        return {"status": "CLASSIFIED", "domain_code": "ROADS_FOOTPATHS", "category_code": "RDW-POT",
                "confidence": 0.85, "evidence": ["pothole", "road"], "explanation": "Identified road issue."}
    elif match_keywords(text_lower, ["garbage", "कचरा"]):
        return {"status": "CLASSIFIED", "domain_code": "SOLID_WASTE", "category_code": "SAN-GARB",
                "confidence": 0.88, "evidence": ["garbage"], "explanation": "Identified solid waste issue."}
    elif match_keywords(text_lower, ["water", "पानी"]):
        return {"status": "CLASSIFIED", "domain_code": "WATER_SUPPLY", "category_code": "WTR-LEAK",
                "confidence": 0.80, "evidence": ["water"], "explanation": "Identified water supply issue."}
    elif match_keywords(text_lower, OOS_KEYWORDS):
        return {"status": "OUT_OF_SCOPE", "domain_code": None, "category_code": None,
                "confidence": 0.95, "evidence": [], "explanation": "Unrelated topic."}
    else:
        # Base model fails on Roman Marathi / Hinglish → hallucinated non-taxonomy code
        return {"status": "CLASSIFIED", "domain_code": "GENERAL", "category_code": "GEN-01",
                "confidence": 0.50, "evidence": ["municipal_issue"],
                "explanation": "Base model uncertain; generated non-taxonomy code."}


def mock_finetuned_inference(text: str, validator: TaxonomyValidator) -> Dict[str, Any]:
    """Simulates FINE-TUNED Qwen2.5-1.5B instruction-tuned inference (understands Roman Marathi)."""
    text_lower = text.lower()

    water_matches = match_keywords(text_lower, WATER_KEYWORDS)
    garbage_matches = match_keywords(text_lower, GARBAGE_KEYWORDS)
    road_matches = match_keywords(text_lower, ROAD_KEYWORDS)
    light_matches = match_keywords(text_lower, LIGHT_KEYWORDS)
    sewer_matches = match_keywords(text_lower, SEWER_KEYWORDS)
    tree_matches = match_keywords(text_lower, TREE_KEYWORDS)
    oos_matches = match_keywords(text_lower, OOS_KEYWORDS)

    if oos_matches:
        return {"status": "OUT_OF_SCOPE", "domain_code": None, "category_code": None,
                "confidence": 0.99, "evidence": oos_matches, "explanation": "Out of scope non-municipal query."}
    elif len(text.strip().split()) < 3:
        return {"status": "INSUFFICIENT_INFORMATION", "domain_code": None, "category_code": None,
                "confidence": 0.92, "evidence": [], "explanation": "Query lacks sufficient detail."}
    elif water_matches:
        return {"status": "CLASSIFIED", "domain_code": "WATER_SUPPLY", "category_code": "WTR-LEAK",
                "confidence": 0.95, "evidence": water_matches,
                "explanation": "Fine-tuned model detected water supply issue from multilingual keywords."}
    elif garbage_matches:
        return {"status": "CLASSIFIED", "domain_code": "SOLID_WASTE", "category_code": "SAN-GARB",
                "confidence": 0.97, "evidence": garbage_matches,
                "explanation": "Fine-tuned model detected solid waste issue from multilingual keywords."}
    elif road_matches:
        return {"status": "CLASSIFIED", "domain_code": "ROADS_FOOTPATHS", "category_code": "RDW-POT",
                "confidence": 0.96, "evidence": road_matches,
                "explanation": "Fine-tuned model detected road issue from multilingual keywords."}
    elif light_matches:
        return {"status": "CLASSIFIED", "domain_code": "STREET_LIGHTING", "category_code": "ELE-LGT",
                "confidence": 0.94, "evidence": light_matches,
                "explanation": "Fine-tuned model detected street lighting issue from multilingual keywords."}
    elif sewer_matches:
        return {"status": "CLASSIFIED", "domain_code": "SEWERAGE_DRAINAGE", "category_code": "SEW-BLOCK",
                "confidence": 0.93, "evidence": sewer_matches,
                "explanation": "Fine-tuned model detected sewerage/drainage issue from multilingual keywords."}
    elif tree_matches:
        return {"status": "CLASSIFIED", "domain_code": "GARDENS_TREES", "category_code": "PRK-TREE",
                "confidence": 0.91, "evidence": tree_matches,
                "explanation": "Fine-tuned model detected garden/tree issue from multilingual keywords."}
    else:
        return {"status": "AMBIGUOUS", "domain_code": None, "category_code": None,
                "confidence": 0.65, "evidence": [],
                "explanation": "Fine-tuned model unable to confidently classify complaint."}


# ============================================================
# OUTPUT FORMATTING & EVALUATION
# ============================================================

def format_model_output(res: Dict[str, Any]) -> str:
    lines = [
        f"status:      {res.get('status')}",
        f"domain:      {res.get('domain_code')}",
        f"category:    {res.get('category_code')}",
        f"confidence:  {res.get('confidence')}",
        f"evidence:    {res.get('evidence')}",
        f"explanation: {res.get('explanation')}"
    ]
    flags = res.get("guardrail_flags", [])
    if flags:
        lines.append(f"guardrails:  {flags}")
    return "\n".join(lines)


def evaluate_single_input(text: str, validator: TaxonomyValidator):
    # 1. Run raw inference
    base_raw = mock_base_inference(text, validator)
    ft_raw = mock_finetuned_inference(text, validator)

    # 2. Apply anti-hallucination guardrails to BOTH outputs
    base_res = apply_guardrails(base_raw, text, validator)
    ft_res = apply_guardrails(ft_raw, text, validator)

    base_tax_valid = validator.is_valid(base_res["status"], base_res["domain_code"], base_res["category_code"])
    ft_tax_valid = validator.is_valid(ft_res["status"], ft_res["domain_code"], ft_res["category_code"])

    agreement = (base_res["status"] == ft_res["status"]) and (base_res["category_code"] == ft_res["category_code"])

    output_str = f"""
INPUT:
{text}

BASE (after guardrails):
{format_model_output(base_res)}

FINE-TUNED (after guardrails):
{format_model_output(ft_res)}

AGREEMENT:
{"YES" if agreement else "NO"}

TAXONOMY VALID:
{"YES" if (base_tax_valid and ft_tax_valid) else f"BASE: {base_tax_valid}, FINE-TUNED: {ft_tax_valid}"}
--------------------------------------------------
"""
    print(output_str)

    # Log result to jsonl
    log_entry = {
        "timestamp": time.time(),
        "input_text": text,
        "base_raw": base_raw,
        "base_after_guardrails": base_res,
        "finetuned_raw": ft_raw,
        "finetuned_after_guardrails": ft_res,
        "agreement": agreement,
        "base_taxonomy_valid": base_tax_valid,
        "finetuned_taxonomy_valid": ft_tax_valid
    }

    MANUAL_TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(MANUAL_TEST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def start_manual_test_mode():
    validator = TaxonomyValidator()
    print("\nMANUAL TEST MODE READY.")
    print("Enter one complaint/keyword set at a time.")
    print("Type EXIT to finish.\n")

    while True:
        try:
            user_input = input("Enter Complaint > ").strip()
            if not user_input:
                continue
            if user_input.upper() == "EXIT":
                print("Exiting Manual Test Mode.")
                break
            evaluate_single_input(user_input, validator)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Manual Test Mode.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--interactive":
        # Non-interactive CLI test mode
        validator = TaxonomyValidator()
        query = " ".join(sys.argv[1:])
        evaluate_single_input(query, validator)
    else:
        start_manual_test_mode()
