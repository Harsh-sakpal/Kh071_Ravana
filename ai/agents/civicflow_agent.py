"""
CivicFlowAgent - The Single Public AI Interface.
Production Entrypoint for Phase 4.

Stateless facade wrapping UnderstandingAgent and ClassificationAgent.
Features:
- Catch-all exception handling.
- Strict taxonomy adherence.
- No Chain-of-Thought in production output.
- Clean JSON trace representation for debugging.
"""

from typing import Dict, Any, Optional
import traceback
from copy import deepcopy

from ai.schemas.dataclasses import (
    ClassificationOutput,
    SemanticClassification,
    AdministrativeMapping,
    ClassificationStatus,
    MappingType,
    ClassificationMethod
)
from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent
from ai.taxonomy.taxonomy_manager import TaxonomyManager

class CivicFlowAgent:
    """Stateless entrypoint for the CivicFlow AI pipeline."""
    
    def __init__(self, taxonomy_manager: Optional[TaxonomyManager] = None):
        # We reuse these component instances, but the process method itself is stateless per request
        self.taxonomy_manager = taxonomy_manager or TaxonomyManager()
        self.understanding_agent = UnderstandingAgent()
        self.classification_agent = ClassificationAgent(taxonomy_manager=self.taxonomy_manager)
        
    def process_complaint(self, text: str, debug: bool = False) -> Dict[str, Any]:
        """
        Process a raw complaint and return a strictly structured payload.
        
        Args:
            text: The raw complaint text from the citizen.
            debug: If True, includes a 'trace' object with execution metadata.
            
        Returns:
            Dict representing the clean, taxonomy-bound result.
        """
        try:
            # 1. Basic validation
            if not text or not isinstance(text, str) or not text.strip():
                return self._build_safe_response(
                    status=ClassificationStatus.INSUFFICIENT_INFORMATION.value,
                    evidence=["Input was empty or strictly whitespace."],
                    explanation="Please provide more details.",
                    debug=debug,
                    trace=self._build_empty_trace()
                )

            # 2. Understanding Agent
            u_out = self.understanding_agent.understand(text)
            
            # Short-circuit if fundamentally malformed/too short
            if u_out.preprocessing_confidence <= 0.5:
                return self._build_safe_response(
                    status=ClassificationStatus.INSUFFICIENT_INFORMATION.value,
                    evidence=u_out.missing_information,
                    explanation="Complaint is too vague or lacks necessary details.",
                    debug=debug,
                    trace=self._build_trace(u_out, None, ClassificationStatus.INSUFFICIENT_INFORMATION.value)
                )

            # 3. Classification Agent
            c_out = self.classification_agent.classify(u_out, debug=debug)
            
            # 4. Taxonomy and Structure Validation
            status = c_out.status
            if status == ClassificationStatus.CLASSIFIED.value:
                dom = c_out.semantic_classification.domain_code
                cat = c_out.semantic_classification.category_code
                
                # Check against TaxonomyManager
                is_valid = False
                if dom and cat:
                    dom_info = self.taxonomy_manager.get_domain(dom)
                    if dom_info and cat in self.taxonomy_manager.get_categories(dom):
                        is_valid = True
                        
                if not is_valid:
                    # Taxonomy violation fallback
                    status = ClassificationStatus.UNKNOWN.value
                    c_out.semantic_classification = SemanticClassification()
                    c_out.administrative_mapping = AdministrativeMapping(type=MappingType.UNMAPPED.value)
                    c_out.evidence.append(f"Internal fallback: Predicted taxonomy {dom}->{cat} was invalid.")
            
            # 5. Build Final Response
            trace_obj = self._build_trace(u_out, c_out, status) if debug else None
            
            # Ensure no chain of thought or raw LLM output leaks into explanation
            explanation = "Classified based on semantic match."
            if status == ClassificationStatus.OUT_OF_SCOPE.value:
                explanation = "Complaint is unrelated to municipal civic duties."
            elif status == ClassificationStatus.AMBIGUOUS.value:
                explanation = "Multiple conflicting issues or vague statements detected."
            elif status == ClassificationStatus.INSUFFICIENT_INFORMATION.value:
                explanation = "Requires more information to classify."
            elif status == ClassificationStatus.UNKNOWN.value:
                explanation = "Unable to classify complaint."
                
            # Filter evidence for production
            evidence = [str(e) for e in c_out.evidence]
            if len(evidence) > 3:
                evidence = evidence[:3]

            result = {
                "status": status,
                "domain_code": c_out.semantic_classification.domain_code,
                "domain_name": c_out.semantic_classification.domain_name,
                "category_code": c_out.semantic_classification.category_code,
                "category_name": c_out.semantic_classification.category_name,
                "department_code": c_out.administrative_mapping.department_code,
                "mapping_type": c_out.administrative_mapping.type,
                "confidence": round(c_out.overall_confidence, 4),
                "evidence": evidence,
                "explanation": explanation
            }
            
            if debug and trace_obj:
                result["trace"] = trace_obj
                
            return result
                
        except Exception as e:
            # Fatal Component Failure Catch
            return self._build_safe_response(
                status=ClassificationStatus.UNKNOWN.value,
                evidence=[f"System exception caught: {str(e)}"],
                explanation="An internal error occurred during classification.",
                debug=debug,
                trace=self._build_empty_trace(error=traceback.format_exc())
            )
            
    def _build_safe_response(self, status: str, evidence: list, explanation: str, debug: bool, trace: dict) -> dict:
        """Helper to generate a guaranteed safe response object."""
        result = {
            "status": status,
            "domain_code": None,
            "domain_name": None,
            "category_code": None,
            "category_name": None,
            "department_code": None,
            "mapping_type": MappingType.UNMAPPED.value,
            "confidence": 0.0,
            "evidence": evidence,
            "explanation": explanation
        }
        if debug:
            result["trace"] = trace
        return result
        
    def _build_empty_trace(self, error: str = None) -> dict:
        """Fallback trace for catastrophic failure or empty input."""
        return {
            "language": "UNKNOWN",
            "normalized_text": "",
            "signals": [],
            "safety_status": "SAFE",
            "decision_status": ClassificationStatus.UNKNOWN.value,
            "method": "ERROR",
            "confidence": 0.0,
            "domain": None,
            "category": None,
            "evidence": [],
            "explanation": "Catastrophic failure or empty input.",
            "error_trace": error
        }
        
    def _build_trace(self, u_out: Any, c_out: Any, final_status: str) -> dict:
        """Constructs the unified debug trace structure as requested."""
        safety = "CRITICAL" if any(s.severity == "CRITICAL" for s in getattr(u_out, 'safety_signals', [])) else \
                 ("HIGH" if any(s.severity == "HIGH" for s in getattr(u_out, 'safety_signals', [])) else "SAFE")
                 
        method = getattr(c_out, 'method', "UNKNOWN") if c_out else "UNKNOWN"
        conf = getattr(c_out, 'overall_confidence', 0.0) if c_out else getattr(u_out, 'preprocessing_confidence', 0.0)
        
        sem = getattr(c_out, 'semantic_classification', None)
        domain = getattr(sem, 'domain_code', None) if sem else None
        category = getattr(sem, 'category_code', None) if sem else None
        
        # Signals list processing
        signals = []
        if hasattr(u_out, 'signals') and isinstance(u_out.signals, dict):
            for k, v in u_out.signals.items():
                if isinstance(v, list) and v:
                    signals.extend(v)
                elif isinstance(v, str) and v and v != "NORMAL":
                    signals.append(v)
                    
        return {
            "language": getattr(u_out.language, 'code', "UNKNOWN") if hasattr(u_out, 'language') else "UNKNOWN",
            "normalized_text": getattr(u_out, 'normalized_text', ""),
            "signals": list(set(signals)),
            "safety_status": safety,
            "decision_status": final_status,
            "method": method,
            "confidence": round(conf, 4),
            "domain": domain,
            "category": category,
            "evidence": [str(e) for e in getattr(c_out, 'evidence', [])][:3] if c_out else [],
            "explanation": "Internal Trace Record"
        }
