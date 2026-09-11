# CivicFlow AI - AI/ML/NLP Module (Person 1 Scope)

## Overview & Role Boundary

This module implements the **AI/ML/NLP Subsystem** for **CivicFlow AI**. It strictly adheres to Person 1 (ML/NLP Engineer) scope and does not modify backend API routes, database schemas, router agents, or frontend code owned by Person 2.

---

## Phase 2 Milestone: Dataset + Hierarchical TF-IDF Baseline

In Phase 2, a reproducible synthetic municipal complaint dataset was generated, validated, and split, followed by the training and evaluation of a **Hierarchical TF-IDF + Calibrated Linear SVM Baseline Model**.

### 1. Dataset Statistics & Stratified Splitting

- **Total Category Supervised Samples**: `3,400`
- **Total Categories**: `34` across `15` fixed AI domains (~100 samples per category)
- **Special Status Dataset**: `200` samples (`AMBIGUOUS`, `UNKNOWN`, `INSUFFICIENT_INFORMATION`, `OUT_OF_SCOPE`)
- **Language Distribution**:
  - English: `1,669` (`49.1%`)
  - Hinglish (Latin transliterated Marathi/Hindi): `906` (`26.6%`)
  - Marathi (Devanagari script): `825` (`24.3%`)
- **Split Breakdown**:
  - `Train`: `2,355` samples (`69.3%`)
  - `Validation`: `504` samples (`14.8%`)
  - `Test`: `541` samples (`15.9%`)
- **Leakage Prevention**: Grouped by text structure prior to stratified category sampling.

---

## Hierarchical Classification Architecture

```
Complaint Text
      │
      ▼
Level 1: TfidfVectorizer + Calibrated LinearSVC (Domain Classifier)
      │
      ▼ (Predicts 1 of 15 AI Domains)
      │
Level 2: TfidfVectorizer + Domain-Specific Calibrated LinearSVC (Category Classifier)
      │
      ▼ (Predicts Category belonging strictly to the predicted domain)
      │
      ▼
Output Contract (Domain + Category + Calibrated Confidence + Top-3 Predictions + Administrative Mapping)
```

---

## Measured Performance Metrics (Phase 2 Test Set Evaluation)

- **Domain Level Accuracy (15 AI Domains)**: `100.00%`
- **Domain Level Macro F1-Score**: `1.0000`
- **Category Level Top-1 Accuracy (34 Categories)**: `100.00%`
- **Category Level Top-3 Accuracy**: `100.00%`
- **Category Level Macro F1-Score**: `1.0000`
- **Category Level Weighted F1-Score**: `1.0000`

*Results saved in `ai/evaluation/results/evaluation_metrics.json` and `ai/evaluation/results/human_readable_report.md`.*

---

## Phase 2 Commands & Tools

### 1. Dataset Generation & Stratified Splitting
```bash
py ai/data/prepare_dataset.py
```

### 2. Model Training & Serialization
```bash
py ai/models/tfidf_model.py
```
*Artifacts saved under `ai/models/tfidf/`.*

### 3. Model Evaluation & Error Analysis
```bash
py ai/evaluation/evaluate_tfidf.py
```

### 4. Interactive Inference Demo
```bash
py ai/tools/demo_inference.py
```

### 5. Automated Unit Test Suite (22 Tests)
```bash
py -m unittest discover -s tests -p "test_*.py"
```

---

## Model Artifacts (`ai/models/tfidf/`)

```
ai/models/tfidf/
├── domain_vectorizer.joblib
├── domain_classifier.joblib
├── metadata.json
├── category_vectorizers/
│   ├── SOLID_WASTE_vectorizer.joblib
│   ├── WATER_SUPPLY_vectorizer.joblib
│   └── ...
└── category_classifiers/
    ├── SOLID_WASTE_classifier.joblib
    ├── WATER_SUPPLY_classifier.joblib
    └── ...
```

---

## Scope & Limitations Notice

- **Phase 2 Status**: TF-IDF Baseline model fully trained, serialized, tested, and evaluated.
- **Explicit Exclusions**: Multilingual vector embeddings, LLM fallback, and agentic confidence routing are **NOT** implemented in Phase 2 (reserved for subsequent phases).
