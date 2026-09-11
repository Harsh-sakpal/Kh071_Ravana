"""
Multilingual Sentence Embedding Model & Category Prototype Indexer for CivicFlow AI.

Model: paraphrase-multilingual-MiniLM-L12-v2
Architecture:
- Level 1: Cosine similarity against 15 AI Domain Centroid Prototypes.
- Level 2: Cosine similarity against Category Centroid Prototypes within predicted Domain.

Saves normalized prototype vectors and metadata to ai/models/embeddings/.
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class HierarchicalEmbeddingModel:
    """Multilingual Sentence Embedding Model with Hierarchical Category Prototype Index."""

    DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, artifacts_dir: Optional[Path] = None):
        self.model_name = model_name
        self.artifacts_dir = artifacts_dir or (Path(__file__).parent / "embeddings")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoder: Optional[Any] = None
        self.category_prototypes: Dict[str, np.ndarray] = {}
        self.domain_prototypes: Dict[str, np.ndarray] = {}
        self.domain_categories: Dict[str, List[str]] = {}
        self.is_loaded = False

    def load_encoder(self) -> None:
        """Loads sentence-transformers model if available."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers package is required for embedding classification. "
                "Please run: pip install sentence-transformers"
            )
        if self.encoder is None:
            print(f"[Embedding Model] Loading {self.model_name}...")
            self.encoder = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes text list into normalized L2 sentence embeddings."""
        self.load_encoder()
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def train_prototypes(self, train_csv_path: str) -> Dict[str, Any]:
        """Reads train CSV, computes category and domain centroids, and serializes prototypes."""
        print(f"[Embedding Index] Reading training examples from {train_csv_path}...")
        cat_texts = defaultdict(list)
        cat_to_domain = {}

        with open(train_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["complaint_text"].strip()
                cat = row["category_code"].strip() if row.get("category_code") else None
                dom = row["domain_code"].strip() if row.get("domain_code") else None

                if text and cat and dom:
                    cat_texts[cat].append(text)
                    cat_to_domain[cat] = dom

        print(f"[Embedding Index] Building prototypes for {len(cat_texts)} categories...")
        cat_vectors = {}
        domain_vectors_accum = defaultdict(list)
        self.domain_categories = defaultdict(list)

        for cat_code, texts in cat_texts.items():
            dom_code = cat_to_domain[cat_code]
            self.domain_categories[dom_code].append(cat_code)

            # Sample top 20 representative texts for fast centroid computation
            sample_texts = texts[:20] if len(texts) > 20 else texts
            vecs = self.encode(sample_texts)
            centroid = np.mean(vecs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            
            cat_vectors[cat_code] = centroid
            domain_vectors_accum[dom_code].append(centroid)

        # Compute domain centroids
        domain_vectors = {}
        for dom_code, cat_vec_list in domain_vectors_accum.items():
            dom_centroid = np.mean(cat_vec_list, axis=0)
            norm = np.linalg.norm(dom_centroid)
            if norm > 0:
                dom_centroid = dom_centroid / norm
            domain_vectors[dom_code] = dom_centroid

        self.category_prototypes = cat_vectors
        self.domain_prototypes = domain_vectors
        self.domain_categories = dict(self.domain_categories)
        self.is_loaded = True

        self.save_artifacts()
        return {
            "status": "INDEXED",
            "total_categories": len(cat_vectors),
            "total_domains": len(domain_vectors),
            "model_name": self.model_name
        }

    def save_artifacts(self) -> None:
        """Saves category and domain prototype vectors and metadata."""
        print(f"[Embedding Index Serialization] Saving artifacts to {self.artifacts_dir}...")
        
        # Save as npz dicts
        np.savez_compressed(self.artifacts_dir / "category_prototypes.npz", **self.category_prototypes)
        np.savez_compressed(self.artifacts_dir / "domain_prototypes.npz", **self.domain_prototypes)

        meta = {
            "model_name": self.model_name,
            "domain_categories": self.domain_categories,
            "categories": list(self.category_prototypes.keys()),
            "domains": list(self.domain_prototypes.keys()),
        }
        with open(self.artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_artifacts(self) -> bool:
        """Loads prototype vectors and metadata from disk."""
        cat_npz = self.artifacts_dir / "category_prototypes.npz"
        dom_npz = self.artifacts_dir / "domain_prototypes.npz"
        meta_p = self.artifacts_dir / "metadata.json"

        if not (cat_npz.exists() and dom_npz.exists() and meta_p.exists()):
            return False

        cat_data = np.load(cat_npz)
        dom_data = np.load(dom_npz)

        self.category_prototypes = {k: cat_data[k] for k in cat_data.files}
        self.domain_prototypes = {k: dom_data[k] for k in dom_data.files}

        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.model_name = meta.get("model_name", self.DEFAULT_MODEL_NAME)
            self.domain_categories = meta.get("domain_categories", {})

        self.is_loaded = True
        return True

    def predict(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Runs hierarchical cosine similarity search.
        Level 1 -> Domain similarity search
        Level 2 -> Category similarity search within predicted domain
        """
        if not self.is_loaded:
            if not self.load_artifacts():
                raise RuntimeError("Embedding prototype index artifacts are not loaded or built.")

        text_vec = self.encode([text])[0]  # Shape (D,)

        # Level 1: Domain Cosine Similarity
        dom_sims = []
        for dom_code, dom_vec in self.domain_prototypes.items():
            sim = float(np.dot(text_vec, dom_vec))
            dom_sims.append((dom_code, sim))

        dom_sims.sort(key=lambda x: x[1], reverse=True)
        pred_domain, top_dom_sim = dom_sims[0]

        # Level 2: Category Cosine Similarity within predicted domain
        candidate_cats = self.domain_categories.get(pred_domain, [])
        cat_sims = []

        for cat_code in candidate_cats:
            cat_vec = self.category_prototypes.get(cat_code)
            if cat_vec is not None:
                sim = float(np.dot(text_vec, cat_vec))
                cat_sims.append((cat_code, sim))

        if not cat_sims:
            # Fallback across all categories if none in domain
            for cat_code, cat_vec in self.category_prototypes.items():
                sim = float(np.dot(text_vec, cat_vec))
                cat_sims.append((cat_code, sim))

        cat_sims.sort(key=lambda x: x[1], reverse=True)
        pred_category, top_cat_sim = cat_sims[0]

        # Calculate score margin between top-1 and top-2
        margin = 0.0
        if len(cat_sims) > 1:
            margin = top_cat_sim - cat_sims[1][1]

        # Documented confidence policy: confidence = top1_sim * (1 + margin) capped at 0.99
        calibrated_conf = round(min(0.99, max(0.0, top_cat_sim * (1.0 + margin))), 4)

        top_predictions = [
            {"code": code, "score": round(sim, 4)}
            for code, sim in cat_sims[:top_k]
        ]

        return {
            "domain": {"code": pred_domain, "score": round(top_dom_sim, 4)},
            "category": {"code": pred_category, "score": round(top_cat_sim, 4)},
            "raw_similarity": round(top_cat_sim, 4),
            "margin": round(margin, 4),
            "overall_confidence": calibrated_conf,
            "top_predictions": top_predictions,
            "method": "EMBEDDING",
        }


if __name__ == "__main__":
    train_csv = Path(__file__).parent.parent / "data" / "processed" / "train.csv"
    model = HierarchicalEmbeddingModel()
    res = model.train_prototypes(str(train_csv))
    print("Embedding prototype indexing result:", res)
