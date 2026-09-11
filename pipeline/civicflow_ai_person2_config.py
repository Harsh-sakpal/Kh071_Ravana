"""
Import bridge for Person 2 taxonomy/config functions.
Centralizes the sys.path manipulation needed for cross-package imports.
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_p2_root = os.path.join(_project_root, "civicflow-ai-person2")

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _p2_root not in sys.path:
    sys.path.insert(0, _p2_root)

from app.config.taxonomy import normalize_domain_name, DOMAIN_ALIASES, REVERSE_ALIASES, TAXONOMY
from app.config.taxonomy import all_domains, all_subcategories, is_valid_domain, is_valid_subcategory

__all__ = [
    "normalize_domain_name",
    "DOMAIN_ALIASES",
    "REVERSE_ALIASES",
    "TAXONOMY",
    "all_domains",
    "all_subcategories",
    "is_valid_domain",
    "is_valid_subcategory",
]
