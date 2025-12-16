from typing import Dict, Any
from .curricular_profile import CurricularProfile

class EpistemologicalValidator:
    def __init__(self):
        self.profile = None

    def configure_curriculum(self, profile: CurricularProfile):
        self.profile = profile

    def validate_concept(self, concept: str) -> Dict[str, Any]:
        if not self.profile:
            raise RuntimeError("Validator not configured with a curriculum profile.")

        # Normalize concept for matching
        concept_lower = concept.lower()
        
        # Check forbidden topics
        for forbidden in self.profile.forbidden_topics:
            if forbidden in concept_lower:
                return {
                    "status": "REJECTED",
                    "reason": "OUT_OF_SCOPE",
                    "confidence": 1.00,
                    "pedagogical_redirect": self._pedagogical_redirect(forbidden)
                }

        # Check for Axiomatic Violations (The Pedagogical Core)
        # We iterate over axioms in the profile to find known violation patterns
        for axiom_name, axiom_data in self.profile.axioms.items():
            for violation in axiom_data.get('common_violations', []):
                 # Normalized check
                 if self._normalize(violation) in self._normalize(concept):
                     return {
                        "status": "AXIOM_VIOLATION",
                        "reason": "STRUCTURAL_ERROR",
                        "confidence": 1.00,
                        "pedagogical_redirect": f"{axiom_data['pedagogical_response']}\n(Axioma violado: {axiom_name})"
                     }

        # Check allowed topics (Simplified keyword matching for pilot)
        # In a real system, this would trace the component parts of the derivation.
        # Here we assume 'concept' is a keyword or question intent.
        
        # We check if any allowed topic keyword is seemingly present or if it's broadly math
        # For strictness, if it's explicitly forbidden, we caught it above.
        # If it's not forbidden, we verify if it matches an allowed topic key.
        
        # Checking against flattened allowed topics
        is_allowed = False
        for allowed in self.profile.allowed_topics:
            # We check if the allowed topic is in the concept query 
            # OR if the query is in the allowed topic (e.g. "polynomial" matches "algebra.polynomials")
            clean_allowed = allowed.split(".")[-1].replace("_", " ")
            if clean_allowed in concept_lower or concept_lower in clean_allowed:
                is_allowed = True
                break
        
        if not is_allowed:
             return {
                "status": "UNKNOWN",
                "reason": "NOT_IN_SYLLABUS",
                "message": "Este concepto no forma parte del programa actual."
            }

        return {"status": "ALLOWED"}

    def _normalize(self, text: str) -> str:
        # Simple normalization: lower case, strip
        return text.lower().strip()

    def _pedagogical_redirect(self, concept: str) -> str:
        # Check if there is a specific redirect, otherwise use generic
        msg = self.profile.redirections.get(concept, self.profile.redirections.get("*", ""))
        
        # Append specific pedagogical framing
        return (
            f"{msg}\n"
            f"(Detectado: '{concept}' - Fuera del Nivel {self.profile.grade_level}º)"
        )
