# CivicFlow AI - Dataset Documentation (Phase 2)

## Overview

This directory contains synthetic raw datasets and processed train/val/test splits for the 15-Domain Municipal Complaint System.

## File Breakdown

```
ai/data/
├── raw/
│   ├── synthetic_complaints.json   # 3,400 category complaints (100 per category)
│   └── special_cases.json          # 200 special cases (AMBIGUOUS, UNKNOWN, INSUFFICIENT_INFORMATION, OUT_OF_SCOPE)
├── processed/
│   ├── train.csv                   # 2,355 training samples (70%)
│   ├── validation.csv              # 504 validation samples (15%)
│   ├── test.csv                    # 541 test samples (15%)
│   └── special_cases.csv           # 200 special cases (kept separate from supervised training)
├── generator.py                    # Synthetic dataset generator engine
├── prepare_dataset.py              # Validation & stratified splitter script
└── README.md
```

## Dataset Schema

Each row in `train.csv`, `validation.csv`, and `test.csv` contains:
- `id`: Unique identifier (e.g. `CF-SYN-1001`)
- `complaint_text`: Raw complaint text
- `domain_code`: One of 15 fixed AI domain codes
- `category_code`: One of 34 taxonomy category codes
- `department_code`: Preserved department code or empty string if unmapped
- `language`: `EN`, `MARATHI`, `HINGLISH`
- `difficulty`: `EASY`, `MEDIUM`, `HARD_OVERLAPPING`
- `source`: `synthetic_generator_v1`

## Leak Prevention & Split Methodology

- Stratified splitting by `category_code`.
- Identical text structures are grouped prior to splitting to ensure 0% data leakage between `train.csv` and `test.csv`.
- Fixed seed (`42`) used for full reproducibility.
