"""
Dataclasses representing structured outputs for Understanding Agent and Classification Agent contracts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class LanguageCode(str, Enum):
    EN = "EN"
    MARATHI = "MARATHI"
    HINGLISH = "HINGLISH"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class ClassificationStatus(str, Enum):
    CLASSIFIED = "CLASSIFIED"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


class MappingType(str, Enum):
    DIRECT = "DIRECT"
    UNMAPPED = "UNMAPPED"
    CROSS_AGENCY = "CROSS_AGENCY"


class ClassificationMethod(str, Enum):
    TFIDF = "TFIDF"
    EMBEDDING = "EMBEDDING"
    LLM_FALLBACK = "LLM_FALLBACK"
    RULE_ENGINE = "RULE_ENGINE"
    HYBRID = "HYBRID"
    NONE = "NONE"


@dataclass
class LanguageInfo:
    code: str = LanguageCode.UNKNOWN.value
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class SafetySignal:
    type: str
    description: str
    severity: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class UnderstandingOutput:
    original_text: str
    normalized_text: str
    language: LanguageInfo
    signals: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, Any] = field(default_factory=dict)
    safety_signals: List[SafetySignal] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    preprocessing_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "language": self.language.to_dict(),
            "signals": self.signals,
            "entities": self.entities,
            "safety_signals": [s.to_dict() for s in self.safety_signals],
            "missing_information": self.missing_information,
            "preprocessing_confidence": round(self.preprocessing_confidence, 4),
        }


@dataclass
class SemanticClassification:
    domain_code: Optional[str] = None
    domain_name: Optional[str] = None
    category_code: Optional[str] = None
    category_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_code": self.domain_code,
            "domain_name": self.domain_name,
            "category_code": self.category_code,
            "category_name": self.category_name,
        }


@dataclass
class AdministrativeMapping:
    type: str = MappingType.DIRECT.value
    department_code: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "department_code": self.department_code,
            "notes": self.notes,
        }


@dataclass
class ClassificationOutput:
    status: str = ClassificationStatus.UNKNOWN.value
    semantic_classification: SemanticClassification = field(default_factory=SemanticClassification)
    administrative_mapping: AdministrativeMapping = field(default_factory=AdministrativeMapping)
    overall_confidence: float = 0.0
    method: str = ClassificationMethod.NONE.value
    evidence: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    secondary_signals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "semantic_classification": self.semantic_classification.to_dict(),
            "administrative_mapping": self.administrative_mapping.to_dict(),
            "overall_confidence": round(self.overall_confidence, 4),
            "method": self.method,
            "evidence": self.evidence,
            "explanation": self.explanation,
            "secondary_signals": self.secondary_signals,
        }
