"""
Extraction, Validation, and Processing Engine for Chat-Generated Synthetic Augmentation Data (Phase 4.1).

Strict Scope: Person 1 ML/NLP only.
Single Source of Truth: ai/taxonomy/taxonomy.json
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.stdout.reconfigure(encoding="utf-8")

# Ensure root workspace in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.taxonomy.taxonomy_manager import TaxonomyManager


class ChatAugmentationProcessor:
    """Processes synthetic dataset pasted in conversation context."""

    def __init__(self):
        self.tm = TaxonomyManager()
        self.raw_dir = Path(__file__).parent / "raw"
        self.processed_dir = Path(__file__).parent / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_transcript(self, transcript_path: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extracts pasted pipe-separated records from conversation transcript log."""
        if transcript_path is None:
            # Default transcript location
            home_dir = Path.home()
            transcript_path = str(home_dir / ".gemini" / "antigravity" / "brain" / "88ff1efd-f6fb-49e8-8aba-6952c46a6593" / ".system_generated" / "logs" / "transcript.jsonl")

        raw_records = []
        stats = {
            "total_extracted_lines": 0,
            "malformed_count": 0,
            "valid_raw_count": 0,
            "malformed_reasons": []
        }

        if not os.path.exists(transcript_path):
            print(f"[Warning] Transcript file not found at {transcript_path}")
            return raw_records, stats

        with open(transcript_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if data.get("type") != "USER_INPUT":
                        continue
                    content = data.get("content", "")
                    if "CIVICFLOW AI — CHAT-GENERATED TRAINING/AUGMENTATION DATA" in content or "chat_generated_augmentation" in content:
                        for row_text in content.split("\n"):
                            row_str = row_text.strip()
                            if "|" in row_str and "complaint_text" not in row_str and "Columns:" not in row_str:
                                stats["total_extracted_lines"] += 1
                                parts = [p.strip() for p in row_str.split("|")]
                                if len(parts) == 6:
                                    text, lang, dom, cat, status, src = parts
                                    # Check for truncation or incomplete tokens
                                    if not text or not cat or len(cat) < 6 or cat not in self.tm.get_categories():
                                        stats["malformed_count"] += 1
                                        stats["malformed_reasons"].append(f"Invalid/Truncated category '{cat}': {row_str}")
                                    else:
                                        raw_records.append({
                                            "complaint_text": text,
                                            "language": lang,
                                            "domain_code": dom,
                                            "category_code": cat,
                                            "status": status,
                                            "source": "synthetic_chat_augmentation",
                                            "raw_line": row_str
                                        })
                                        stats["valid_raw_count"] += 1
                                else:
                                    stats["malformed_count"] += 1
                                    stats["malformed_reasons"].append(f"Incorrect column count ({len(parts)}): {row_str}")
                except Exception:
                    continue

        return raw_records, stats

    def validate_taxonomy(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validates domain, category, and status against taxonomy.json single source of truth."""
        valid_records = []
        stats = {
            "valid_taxonomy_count": 0,
            "invalid_taxonomy_count": 0,
            "invalid_domain_codes": [],
            "invalid_category_codes": [],
            "mismatch_count": 0,
            "invalid_details": []
        }

        valid_statuses = {"CLASSIFIED", "AMBIGUOUS", "UNKNOWN", "OUT_OF_SCOPE", "INSUFFICIENT_INFORMATION"}

        for r in records:
            dom = r.get("domain_code")
            cat = r.get("category_code")
            status = r.get("status")

            dom_valid = self.tm.get_domain(dom) is not None
            cat_valid = self.tm.get_category(cat) is not None
            status_valid = status in valid_statuses

            # Domain/Category relationship check
            cat_info = self.tm.get_category(cat) if cat_valid else None
            domain_match = False
            if cat_info and dom_valid:
                domain_categories = self.tm.data.get("domains", {}).get(dom, {}).get("categories", {})
                domain_match = cat in domain_categories

            if dom_valid and cat_valid and status_valid and domain_match:
                valid_records.append(r)
                stats["valid_taxonomy_count"] += 1
            else:
                stats["invalid_taxonomy_count"] += 1
                reason = []
                if not dom_valid:
                    stats["invalid_domain_codes"].append(dom)
                    reason.append(f"Invalid domain '{dom}'")
                if not cat_valid:
                    stats["invalid_category_codes"].append(cat)
                    reason.append(f"Invalid category '{cat}'")
                if not status_valid:
                    reason.append(f"Invalid status '{status}'")
                if dom_valid and cat_valid and not domain_match:
                    stats["mismatch_count"] += 1
                    reason.append(f"Category '{cat}' does not belong to domain '{dom}'")

                stats["invalid_details"].append({
                    "record": r,
                    "reason": "; ".join(reason)
                })

        return valid_records, stats

    def check_leakage(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Checks for text leakage against existing splits (train, validation, test, challenge, uncertainty)."""
        evaluation_splits = {
            "train": self.processed_dir / "train.csv",
            "validation": self.processed_dir / "validation.csv",
            "test": self.processed_dir / "test.csv",
            "challenge": self.processed_dir / "challenge_test.csv",
            "uncertainty": self.processed_dir / "uncertainty_test.csv",
        }

        existing_texts = {}
        for split_name, path in evaluation_splits.items():
            existing_texts[split_name] = set()
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        txt = row.get("complaint_text", "").strip().lower()
                        if txt:
                            existing_texts[split_name].add(txt)

        clean_records = []
        leakage_stats = {
            "total_checked": len(records),
            "overlap_train": 0,
            "overlap_validation": 0,
            "overlap_test": 0,
            "overlap_challenge": 0,
            "overlap_uncertainty": 0,
            "removed_for_leakage": 0,
            "clean_count": 0,
        }

        for r in records:
            norm_txt = r["complaint_text"].strip().lower()
            is_leaked = False

            if norm_txt in existing_texts["train"]:
                leakage_stats["overlap_train"] += 1
            if norm_txt in existing_texts["validation"]:
                leakage_stats["overlap_validation"] += 1
                is_leaked = True
            if norm_txt in existing_texts["test"]:
                leakage_stats["overlap_test"] += 1
                is_leaked = True
            if norm_txt in existing_texts["challenge"]:
                leakage_stats["overlap_challenge"] += 1
                is_leaked = True
            if norm_txt in existing_texts["uncertainty"]:
                leakage_stats["overlap_uncertainty"] += 1
                is_leaked = True

            if is_leaked:
                leakage_stats["removed_for_leakage"] += 1
            else:
                clean_records.append(r)

        leakage_stats["clean_count"] = len(clean_records)
        return clean_records, leakage_stats

    def save_clean_datasets(self, clean_records: List[Dict[str, Any]]) -> Path:
        """Saves clean augmentation records to ai/data/raw/chat_training_data_clean.csv."""
        output_path = self.raw_dir / "chat_training_data_clean.csv"
        fieldnames = ["complaint_text", "language", "domain_code", "category_code", "status", "source"]

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in clean_records:
                writer.writerow({
                    "complaint_text": r["complaint_text"],
                    "language": r["language"],
                    "domain_code": r["domain_code"],
                    "category_code": r["category_code"],
                    "status": r["status"],
                    "source": "synthetic_chat_augmentation",
                })

        print(f"[Augmentation Processor] Saved {len(clean_records)} clean augmentation records to {output_path}")
        return output_path


if __name__ == "__main__":
    processor = ChatAugmentationProcessor()
    raw_records, stats = processor.extract_from_transcript()
    print("Extracted Records Stats:", stats)
    valid_records, tax_stats = processor.validate_taxonomy(raw_records)
    print("Taxonomy Validation Stats:", tax_stats)
    clean_records, leak_stats = processor.check_leakage(valid_records)
    print("Leakage Check Stats:", leak_stats)
    if clean_records:
        processor.save_clean_datasets(clean_records)
