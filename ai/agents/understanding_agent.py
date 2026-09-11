"""
Understanding Agent implementation for CivicFlow AI (Person 1 Scope).

Interprets and structures raw citizen complaints into structured understanding outputs,
preserving original text, safely normalizing language, detecting languages (EN, Marathi, Hinglish, Mixed),
extracting signals, entities, safety hazard alerts, and identifying missing information.
"""

import re
from typing import Dict, Any, List, Tuple
from ai.schemas.dataclasses import (
    UnderstandingOutput,
    LanguageInfo,
    LanguageCode,
    SafetySignal,
)


class LanguageDetector:
    """Pluggable language detector for English, Marathi (Devanagari), Hinglish, and Mixed text."""

    DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
    
    MARATHI_LATIN_KEYWORDS = {
        "aahe", "aala", "aale", "zala", "zali", "zale", "rastyavar", "rastyavarchi", "rastyavarchya",
        "kachra", "paanyachi", "paani", "pani", "gatar", "gatarrache", "motha", "khadda", "dukan",
        "tutla", "tutli", "ughade", "bhinta", "imaratichi", "dhoka", "phatla", "baghichi", "yetahe",
        "panayache", "sathle", "padla", "padli", "machhar", "sarvajanik", "shauchalay", "ghanerde",
        "vadhle", "khup", "maagchya", "aathavdya", "pasun", "chalat", "nahiye", "amchya", "gallyat",
        "misalle", "nallhyache", "aani", "sachalay"
    }

    HINGLISH_KEYWORDS = {
        "hai", "raha", "rahi", "ho", "gaya", "gayi", "bohot", "bahut", "bhi",
        "khula", "par", "rasta", "bhai", "samne", "karamchari", "paani", "kachra",
        "nahi", "liye", "kya", "kar"
    }

    def detect(self, text: str) -> LanguageInfo:
        """Detects language of the input text."""
        if not text or not text.strip():
            return LanguageInfo(code=LanguageCode.UNKNOWN.value, confidence=0.0)

        # Count Devanagari characters
        devanagari_chars = len(self.DEVANAGARI_REGEX.findall(text))
        total_letters = sum(1 for c in text if c.isalpha())

        if total_letters == 0:
            return LanguageInfo(code=LanguageCode.UNKNOWN.value, confidence=0.5)

        devanagari_ratio = devanagari_chars / max(total_letters, 1)

        words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))

        marathi_latin_hits = len(words.intersection(self.MARATHI_LATIN_KEYWORDS))
        hinglish_hits = len(words.intersection(self.HINGLISH_KEYWORDS))

        # Check stem/substring match for Marathi transliteration
        if marathi_latin_hits == 0:
            for w in words:
                if any(kw in w for kw in ["rastyav", "gatar", "kachra", "nahiye", "aathav", "chalat", "paany"]):
                    marathi_latin_hits += 1

        if devanagari_ratio > 0.4:
            if len(words) > 2 and devanagari_ratio < 0.8:
                return LanguageInfo(code=LanguageCode.MIXED.value, confidence=0.85)
            return LanguageInfo(code=LanguageCode.MARATHI.value, confidence=0.95)

        if marathi_latin_hits >= 1 or hinglish_hits >= 1:
            confidence = min(0.70 + (marathi_latin_hits + hinglish_hits) * 0.1, 0.98)
            return LanguageInfo(code=LanguageCode.HINGLISH.value, confidence=confidence)

        # Default to English if Latin script without regional markers
        return LanguageInfo(code=LanguageCode.EN.value, confidence=0.90)


class UnderstandingAgent:
    """
    Understanding Agent prepares structured input for the Classification Agent.
    Does NOT perform final domain/category classification.
    """

    def __init__(self, language_detector: float = None):
        self.language_detector = LanguageDetector()

    def normalize_text(self, text: str) -> str:
        """
        Safely normalizes text without destroying complaint meaning.
        Preserves Devanagari script, Latin text, numbers, and key punctuation.
        """
        if not text:
            return ""
        # Lowercase
        normalized = text.lower()
        # Collapse multi-spaces and newlines
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def extract_safety_signals(self, text: str, normalized_text: str) -> List[SafetySignal]:
        """Extracts critical civic safety signals (electrical hazards, open manholes, collapse risk)."""
        safety_signals = []

        # Open manholes / Gatar
        if re.search(r"\b(open manhole|missing cover|ughade gatar|open chamber|broken chamber)\b", normalized_text):
            safety_signals.append(
                SafetySignal(
                    type="OPEN_MANHOLE_HAZARD",
                    description="Open or missing manhole cover detected - physical fall hazard",
                    severity="HIGH",
                )
            )

        # Electrical hazards / Live wires
        if re.search(r"\b(live wire|exposed wire|hanging wire|sparking|shock|ughadi wire|vidyut khamba)\b", normalized_text):
            safety_signals.append(
                SafetySignal(
                    type="ELECTRICAL_HAZARD",
                    description="Exposed live wire or dangerous electrical pole hazard detected",
                    severity="CRITICAL",
                )
            )

        # Structural collapse / Fire hazard
        if re.search(r"\b(collapse|dilapidated|cracked wall|padnyacha dhoka|fire exit|flammable)\b", normalized_text):
            safety_signals.append(
                SafetySignal(
                    type="STRUCTURAL_FIRE_HAZARD",
                    description="Structural integrity or fire safety emergency hazard detected",
                    severity="HIGH",
                )
            )

        # Waterlogging flooding
        if re.search(r"\b(knee deep|submerged|flooding|waterlogging)\b", normalized_text):
            safety_signals.append(
                SafetySignal(
                    type="FLOODING_HAZARD",
                    description="Severe waterlogging or localized flooding detected",
                    severity="MEDIUM",
                )
            )

        return safety_signals

    def extract_signals(self, normalized_text: str) -> Dict[str, Any]:
        """Extracts material, urgency, and infrastructure attribute signals."""
        signals = {
            "urgency": "NORMAL",
            "material_attributes": [],
            "infrastructure_types": [],
        }

        # Urgency markers
        if re.search(r"\b(urgent|emergency|immediately|critical|turt|lagaatar|motha)\b", normalized_text):
            signals["urgency"] = "HIGH"

        # Material attributes (preserved as signals, NOT categories)
        materials = []
        for mat in ["plastic", "debris", "concrete", "asphalt", "sewage", "mud", "water", "glass", "chemical"]:
            if mat in normalized_text:
                materials.append(mat)
        signals["material_attributes"] = materials

        # Infrastructure types
        infra = []
        for item in ["footpath", "road", "street", "pipe", "light", "pole", "tree", "drain", "manhole", "bin", "park"]:
            if item in normalized_text:
                infra.append(item)
        signals["infrastructure_types"] = infra

        return signals

    def extract_entities(self, text: str, normalized_text: str) -> Dict[str, Any]:
        """Extracts entity markers such as location cues, landmarks, and duration expressions."""
        entities = {
            "locations": [],
            "landmarks": [],
            "durations": [],
        }

        # Location prepositions & markers
        loc_matches = re.findall(
            r"\b(?:near|outside|at|in|opposite|behind|beside|jawad|samne)\s+([a-zA-Z0-9\s\u0900-\u097F]{3,25})\b",
            normalized_text,
        )
        if loc_matches:
            entities["landmarks"] = [m.strip() for m in loc_matches[:3]]

        # Duration cues
        dur_matches = re.findall(r"\b(\d+\s*(?:days|hours|weeks|months|divas|mahine))\b", normalized_text)
        if dur_matches:
            entities["durations"] = dur_matches

        return entities

    def detect_missing_information(self, text: str, normalized_text: str, entities: Dict[str, Any]) -> List[str]:
        """Identifies missing actionable details (e.g. location details missing)."""
        missing = []
        words = normalized_text.split()

        if len(words) < 4:
            missing.append("INSUFFICIENT_DETAILS: Complaint text is extremely short and vague")

        if not entities.get("landmarks") and not re.search(r"\b(road|street|ward|colony|area|plot|lane|rastyavar|society|block)\b", normalized_text):
            missing.append("MISSING_LOCATION: Specific location, street, or landmark context is missing")

        return missing

    def understand(self, complaint_text: str) -> UnderstandingOutput:
        """
        Main entrypoint: Accepts raw complaint text and produces structured UnderstandingOutput.
        """
        if complaint_text is None:
            complaint_text = ""

        # 1. Preserve original text
        original_text = complaint_text

        # 2. Normalize text safely
        normalized_text = self.normalize_text(original_text)

        # 3. Detect language
        language_info = self.language_detector.detect(original_text)

        # 4. Extract signals & entities
        signals = self.extract_signals(normalized_text)
        entities = self.extract_entities(original_text, normalized_text)

        # 5. Extract safety signals
        safety_signals = self.extract_safety_signals(original_text, normalized_text)

        # 6. Detect missing information
        missing_info = self.detect_missing_information(original_text, normalized_text, entities)

        # 7. Compute preprocessing confidence
        preprocessing_confidence = 1.0
        if not normalized_text:
            preprocessing_confidence = 0.0
        elif missing_info:
            preprocessing_confidence = max(0.5, 1.0 - (0.2 * len(missing_info)))

        return UnderstandingOutput(
            original_text=original_text,
            normalized_text=normalized_text,
            language=language_info,
            signals=signals,
            entities=entities,
            safety_signals=safety_signals,
            missing_information=missing_info,
            preprocessing_confidence=preprocessing_confidence,
        )
