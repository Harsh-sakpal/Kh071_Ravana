# CivicFlow AI - Evaluation Framework & Results

This directory contains evaluation scripts and measured performance reports for the Phase 2 TF-IDF Baseline model.

## Evaluation Results Location

```
ai/evaluation/
├── evaluate_tfidf.py              # Evaluation and error analysis script
├── results/
│   ├── evaluation_metrics.json    # Machine-readable evaluation metrics
│   ├── error_analysis.json        # Categorized misclassification records
│   └── human_readable_report.md   # Markdown summary report
└── README.md
```

## Measured Benchmark Results (Phase 2 Test Set)

| Metric | Domain Level (15 Domains) | Category Level (34 Categories) |
|---|---|---|
| **Top-1 Accuracy** | **100.00%** | **100.00%** |
| **Top-3 Accuracy** | N/A | **100.00%** |
| **Macro Precision** | 1.0000 | 1.0000 |
| **Macro Recall** | 1.0000 | 1.0000 |
| **Macro F1-Score** | **1.0000** | **1.0000** |
| **Weighted F1-Score** | **1.0000** | **1.0000** |
