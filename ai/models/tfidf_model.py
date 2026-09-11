"""
Hierarchical TF-IDF Baseline Model for CivicFlow AI.

Architecture:
- Level 1: Complaint Text -> TfidfVectorizer -> Calibrated LinearSVC Domain Classifier (15 AI Domains)
- Level 2: Complaint Text + Predicted Domain -> Domain-Specific Calibrated LinearSVC Category Classifiers

Provides training, evaluation support, artifact serialization (joblib), and inference routines.
"""

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from ai.data.marathi_lexicon import normalize_marathi_text


class HierarchicalTfidfModel:
    """Hierarchical TF-IDF + Calibrated Linear SVM Model."""

    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or (Path(__file__).parent / "tfidf")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.domain_vectorizer: Optional[TfidfVectorizer] = None
        self.domain_classifier: Optional[CalibratedClassifierCV] = None
        
        # Level 2 classifiers per domain code
        self.category_classifiers: Dict[str, CalibratedClassifierCV] = {}
        self.category_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.domain_categories: Dict[str, List[str]] = {}

    def build_vectorizer(self) -> TfidfVectorizer:
        """Returns configured TfidfVectorizer."""
        return TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=10000,
            sublinear_tf=True,
            strip_accents="unicode",
        )

    def train(self, train_csv_path: str) -> Dict[str, Any]:
        """Trains Level 1 Domain classifier and Level 2 Category classifiers."""
        print(f"[Model Training] Reading training data from {train_csv_path}...")
        texts = []
        domains = []
        categories = []

        with open(train_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["complaint_text"].strip()
                text = normalize_marathi_text(text)
                dom = row["domain_code"].strip() if row.get("domain_code") else None
                cat = row["category_code"].strip() if row.get("category_code") else None

                if text and dom and cat:
                    texts.append(text)
                    domains.append(dom)
                    categories.append(cat)

        print(f"[Model Training] Total valid training samples: {len(texts)}")

        # 1. Train Level 1 Domain Model
        print("[Level 1] Training Domain Vectorizer and Calibrated LinearSVC Classifier...")
        self.domain_vectorizer = self.build_vectorizer()
        X_domain = self.domain_vectorizer.fit_transform(texts)

        base_svc = LinearSVC(C=1.0, random_state=42, dual=False, max_iter=2000)
        self.domain_classifier = CalibratedClassifierCV(estimator=base_svc, cv=3)
        self.domain_classifier.fit(X_domain, domains)

        # Map categories belonging to each domain
        domain_to_data = {}
        for text, dom, cat in zip(texts, domains, categories):
            if dom not in domain_to_data:
                domain_to_data[dom] = {"texts": [], "categories": []}
                self.domain_categories[dom] = set()
            domain_to_data[dom]["texts"].append(text)
            domain_to_data[dom]["categories"].append(cat)
            self.domain_categories[dom].add(cat)

        self.domain_categories = {k: list(v) for k, v in self.domain_categories.items()}

        # 2. Train Level 2 Category Models per Domain
        print("[Level 2] Training per-domain Category Classifiers...")
        for dom, data in domain_to_data.items():
            cat_texts = data["texts"]
            cat_labels = data["categories"]
            unique_cats = list(set(cat_labels))

            if len(unique_cats) <= 1:
                # Single category domain
                self.category_classifiers[dom] = None
                self.category_vectorizers[dom] = None
                print(f"  - Domain {dom}: Single category ({unique_cats[0]}), classifier bypassed.")
                continue

            vec = self.build_vectorizer()
            X_cat = vec.fit_transform(cat_texts)

            # Adjust cv folds based on min category count
            cat_counts = Counter(cat_labels)
            min_count = min(cat_counts.values())
            cv_folds = min(3, max(2, min_count))

            cat_svc = LinearSVC(C=1.0, random_state=42, dual=False, max_iter=2000)
            calibrated_cat = CalibratedClassifierCV(estimator=cat_svc, cv=cv_folds)
            calibrated_cat.fit(X_cat, cat_labels)

            self.category_vectorizers[dom] = vec
            self.category_classifiers[dom] = calibrated_cat
            print(f"  - Domain {dom}: Trained classifier for {len(unique_cats)} categories ({cat_counts}).")

        # Save artifacts
        self.save_artifacts()
        return {
            "status": "TRAINED",
            "total_samples": len(texts),
            "domains_count": len(set(domains)),
            "categories_count": len(set(categories)),
        }

    def save_artifacts(self) -> None:
        """Saves trained model artifacts to artifacts_dir."""
        print(f"[Model Serialization] Saving artifacts to {self.artifacts_dir}...")
        
        joblib.dump(self.domain_vectorizer, self.artifacts_dir / "domain_vectorizer.joblib")
        joblib.dump(self.domain_classifier, self.artifacts_dir / "domain_classifier.joblib")

        cat_cls_dir = self.artifacts_dir / "category_classifiers"
        cat_vec_dir = self.artifacts_dir / "category_vectorizers"
        cat_cls_dir.mkdir(exist_ok=True)
        cat_vec_dir.mkdir(exist_ok=True)

        for dom, cat_cls in self.category_classifiers.items():
            if cat_cls is not None:
                joblib.dump(cat_cls, cat_cls_dir / f"{dom}_classifier.joblib")
            if self.category_vectorizers.get(dom) is not None:
                joblib.dump(self.category_vectorizers[dom], cat_vec_dir / f"{dom}_vectorizer.joblib")

        meta = {
            "domain_categories": self.domain_categories,
            "domains": list(self.domain_classifier.classes_) if self.domain_classifier else [],
        }
        with open(self.artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("[Model Serialization] Saved metadata.json and joblib classifiers.")

    def load_artifacts(self) -> bool:
        """Loads serialized model artifacts from artifacts_dir."""
        dom_vec_path = self.artifacts_dir / "domain_vectorizer.joblib"
        dom_cls_path = self.artifacts_dir / "domain_classifier.joblib"
        meta_path = self.artifacts_dir / "metadata.json"

        if not (dom_vec_path.exists() and dom_cls_path.exists() and meta_path.exists()):
            return False

        self.domain_vectorizer = joblib.load(dom_vec_path)
        self.domain_classifier = joblib.load(dom_cls_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.domain_categories = meta.get("domain_categories", {})

        cat_cls_dir = self.artifacts_dir / "category_classifiers"
        cat_vec_dir = self.artifacts_dir / "category_vectorizers"

        for dom in self.domain_categories:
            cls_p = cat_cls_dir / f"{dom}_classifier.joblib"
            vec_p = cat_vec_dir / f"{dom}_vectorizer.joblib"

            if cls_p.exists() and vec_p.exists():
                self.category_classifiers[dom] = joblib.load(cls_p)
                self.category_vectorizers[dom] = joblib.load(vec_p)
            else:
                self.category_classifiers[dom] = None
                self.category_vectorizers[dom] = None

        return True

    def predict(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Runs hierarchical prediction on input text.
        Level 1 -> Domain prediction with calibrated confidence
        Level 2 -> Category prediction constrained to predicted domain
        """
        if not self.domain_classifier or not self.domain_vectorizer:
            raise RuntimeError("Model is not loaded or trained.")

        if not text or not text.strip():
            return {
                "domain": {"code": None, "confidence": 0.0},
                "category": {"code": None, "confidence": 0.0},
                "overall_confidence": 0.0,
                "top_predictions": [],
                "method": "TFIDF",
            }

        text = normalize_marathi_text(text)

        # Level 1: Domain prediction
        X_dom = self.domain_vectorizer.transform([text])
        dom_probs = self.domain_classifier.predict_proba(X_dom)[0]
        dom_classes = self.domain_classifier.classes_

        top_dom_idx = np.argsort(dom_probs)[::-1]
        pred_domain = dom_classes[top_dom_idx[0]]
        domain_conf = float(dom_probs[top_dom_idx[0]])

        # Level 2: Category prediction within predicted domain
        cat_cls = self.category_classifiers.get(pred_domain)
        cat_vec = self.category_vectorizers.get(pred_domain)
        possible_cats = self.domain_categories.get(pred_domain, [])

        if cat_cls is None or cat_vec is None or len(possible_cats) <= 1:
            pred_category = possible_cats[0] if possible_cats else None
            cat_conf = domain_conf
            top_cat_preds = [{"code": pred_category, "score": round(cat_conf, 4)}]
        else:
            X_cat = cat_vec.transform([text])
            cat_probs = cat_cls.predict_proba(X_cat)[0]
            cat_classes = cat_cls.classes_

            top_cat_idx = np.argsort(cat_probs)[::-1]
            pred_category = cat_classes[top_cat_idx[0]]
            cat_conf = float(cat_probs[top_cat_idx[0]])

            top_cat_preds = []
            for idx in top_cat_idx[:top_k]:
                top_cat_preds.append({
                    "code": cat_classes[idx],
                    "score": round(float(cat_probs[idx]), 4),
                })

        overall_conf = round(domain_conf * cat_conf, 4)

        return {
            "domain": {"code": pred_domain, "confidence": round(domain_conf, 4)},
            "category": {"code": pred_category, "confidence": round(cat_conf, 4)},
            "overall_confidence": overall_conf,
            "top_predictions": top_cat_preds,
            "method": "TFIDF",
        }


if __name__ == "__main__":
    train_csv = Path(__file__).parent.parent / "data" / "processed" / "train.csv"
    model = HierarchicalTfidfModel()
    res = model.train(str(train_csv))
    print("Training result:", res)
