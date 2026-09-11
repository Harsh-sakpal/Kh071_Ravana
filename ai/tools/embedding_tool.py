"""
Embedding Similarity Tool for CivicFlow AI.
Wraps HierarchicalEmbeddingModel providing vector similarity inference to Classification Agent.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from ai.models.embedding_model import HierarchicalEmbeddingModel
from ai.taxonomy.taxonomy_manager import TaxonomyManager


class EmbeddingSimilarityTool:
    """Classification tool using Multilingual Sentence Embeddings and Cosine Prototype Matching."""

    def __init__(self, artifacts_dir: Optional[str] = None, taxonomy_manager: Optional[TaxonomyManager] = None):
        path = Path(artifacts_dir) if artifacts_dir else None
        self.model = HierarchicalEmbeddingModel(artifacts_dir=path)
        self.taxonomy_manager = taxonomy_manager or TaxonomyManager()
        self.is_ready = self.load()

    def load(self) -> bool:
        """Loads prototype artifacts from disk."""
        return self.model.load_artifacts()

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies input complaint text into domain, category, and administrative department mapping
        using semantic vector similarity.
        """
        if not self.is_ready:
            self.is_ready = self.load()
            if not self.is_ready:
                raise RuntimeError("Embedding prototype artifacts are not built or loaded.")

        pred = self.model.predict(text)
        cat_code = pred["category"]["code"]
        cat_info = self.taxonomy_manager.get_category(cat_code) if cat_code else None

        dept_map = None
        if cat_code:
            dept_map = self.taxonomy_manager.get_department_mapping(cat_code)

        status = "CLASSIFIED" if pred["overall_confidence"] >= 0.35 else "AMBIGUOUS"

        category_res = pred["category"].copy()
        if cat_info:
            category_res["name"] = cat_info.get("category_name")

        return {
            "status": status,
            "domain": pred["domain"],
            "category": category_res,
            "administrative_mapping": dept_map or {"type": "UNMAPPED", "department_code": None},
            "overall_confidence": pred["overall_confidence"],
            "raw_similarity": pred["raw_similarity"],
            "margin": pred["margin"],
            "top_predictions": pred["top_predictions"],
            "method": "EMBEDDING"
        }
