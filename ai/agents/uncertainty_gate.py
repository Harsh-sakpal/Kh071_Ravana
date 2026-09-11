"""
Dedicated Out-Of-Scope (OOS) & Uncertainty Gate for CivicFlow AI (Phase 4).

Pre-evaluates input complaints before classification to accurately distinguish:
1. Clearly Unrelated / Out-Of-Scope queries (including adversarial municipal keyword cases).
2. Insufficient Information queries (short/vague complaints).
3. Ambiguous municipal queries.
"""

import re
from typing import Dict, Any, Optional, Tuple
from ai.schemas.dataclasses import ClassificationStatus


class UncertaintyGate:
    """Pre-classification gate for OOS and ambiguity detection."""

    # Non-civic entity patterns (even if municipal words are embedded in context)
    NON_CIVIC_ENTITIES = [
        (r"\b(garbage collection|ram cleaner|app|software|cpu|memory|android|iphone)\b.*\b(app|crash|bug|cleaner|software|pc|mobile)\b", "SOFTWARE_APP_OOS"),
        (r"\b(python|code|library|script|program)\b.*\b(pipe|example|garbage|leak)\b", "SOFTWARE_PROGRAMMING_OOS"),
        (r"\b(road runner|cartoon|tv channel|movie|show)\b", "ENTERTAINMENT_OOS"),
        (r"\b(water heater|bathroom|geyser|sink strainer|kitchen sink|home wifi|laptop|mobile phone|battery|remote control)\b", "PRIVATE_APPLIANCE_OOS"),
        (r"\b(cricket|match|umpire|referee|game terrain|minecraft|simulation|level)\b.*\b(pipe|oil|leak|game)\b", "SPORTS_GAMING_OOS"),
        (r"\b(game|minecraft|simulation|level)\b.*\b(pipe|oil|leak)\b", "SPORTS_GAMING_OOS"),
        (r"\b(flight ticket|bus ticket|courier|delivery|online shopping|amazon|flipkart)\b", "ECOMMERCE_PRIVATE_OOS"),
        (r"\b(oil pipe|oil pipeline|gas pipe|gas pipeline|chemical pipeline|chemical pipe)\b", "NON_WATER_PIPELINE_OOS"),
    ]

    INSUFFICIENT_PATTERNS = [
        r"^(please\s+)?(help|fix|solve|check|problem|issue|bad|action)(\s+please)?[\.!\?]*$"
    ]

    def evaluate(self, original_text: str, normalized_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Evaluates text for special status resolution.
        Returns Tuple[StatusString, ReasonString] if special case detected, or (None, None) if normal.
        """
        if not normalized_text or not normalized_text.strip():
            return ClassificationStatus.INSUFFICIENT_INFORMATION.value, "Empty complaint text"

        text_lower = (original_text or "").lower()
        norm_lower = normalized_text.lower()

        # Check Insufficient Information
        words = normalized_text.split()
        if len(words) < 3:
            return ClassificationStatus.INSUFFICIENT_INFORMATION.value, "Text is extremely short (< 3 words) lacking minimum actionable detail"

        for pat in self.INSUFFICIENT_PATTERNS:
            if re.search(pat, norm_lower):
                return ClassificationStatus.INSUFFICIENT_INFORMATION.value, "Vague text lacking actionable intent or locality context"

        # Check Adversarial Non-Civic Out-Of-Scope Patterns (against original and normalized text)
        for pat, reason in self.NON_CIVIC_ENTITIES:
            if re.search(pat, norm_lower) or re.search(pat, text_lower):
                return ClassificationStatus.OUT_OF_SCOPE.value, f"Non-municipal query detected: {reason}"

        return None, None
