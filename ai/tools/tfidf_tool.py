"""
TF-IDF Classification Tool Interface for CivicFlow AI.
Wraps HierarchicalTfidfModel providing model inference to Classification Agent.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from ai.models.tfidf_model import HierarchicalTfidfModel
from ai.taxonomy.taxonomy_manager import TaxonomyManager


class TfidfClassificationTool:
    """Classification tool using Hierarchical TF-IDF + Calibrated Linear SVM."""

    def __init__(self, artifacts_dir: Optional[str] = None, taxonomy_manager: Optional[TaxonomyManager] = None):
        path = Path(artifacts_dir) if artifacts_dir else None
        self.model = HierarchicalTfidfModel(artifacts_dir=path)
        self.taxonomy_manager = taxonomy_manager or TaxonomyManager()
        self.is_ready = self.load()

    def load(self) -> bool:
        """Loads model artifacts from disk."""
        return self.model.load_artifacts()

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies input complaint text into domain, category, and administrative department mapping.
        Returns structured dictionary.
        """
        if not self.is_ready:
            # Try reloading once
            self.is_ready = self.load()
            if not self.is_ready:
                raise RuntimeError("TF-IDF classification model artifacts are not loaded or trained.")

        pred = self.model.predict(text)
        cat_code = pred["category"]["code"]
        dom_code = pred["domain"]["code"]

        dept_map = None
        if cat_code:
            dept_map = self.taxonomy_manager.get_department_mapping(cat_code)

        status = "CLASSIFIED" if pred["overall_confidence"] >= 0.35 else "AMBIGUOUS"

        return {
            "status": status,
            "domain": pred["domain"],
            "category": pred["category"],
            "administrative_mapping": dept_map or {"type": "UNMAPPED", "department_code": None},
            "overall_confidence": pred["overall_confidence"],
            "top_predictions": pred["top_predictions"],
            "method": "TFIDF"
        }
