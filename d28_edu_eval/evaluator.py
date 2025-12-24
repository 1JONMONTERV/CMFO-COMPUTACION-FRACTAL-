import json
import sys
import os
from typing import Dict, Any, List

sys.path.append(os.getcwd())

class SovereignEvaluator:
    """
    D28 Core: Structural Evaluation Engine.
    Evaluates answers based on 'Axiomatic Compliance', not textual similarity.
    """
    
    def __init__(self, rubrics_path: str):
        with open(rubrics_path, 'r', encoding='utf-8') as f:
            self.rubrics = json.load(f)
        print(f"[*] Sovereign Evaluator Loaded. Domain: {self.rubrics.get('domain')}")

    def evaluate(self, problem_id: str, student_answer: str) -> Dict[str, Any]:
        """
        Diagnoses a student answer.
        Returns: { status, feedback, axiom_ref, missing_terms }
        """
        if problem_id not in self.rubrics["problems"]:
            return {"status": "ERROR", "feedback": "Problem ID not found in rubric."}
            
        rubric = self.rubrics["problems"][problem_id]
        answer_norm = self._normalize(student_answer)
        
        # 1. Check for Specific Violations (Traps)
        # We check these FIRST because they represent specific misconceptions we want to catch.
        for violation_id, v_data in rubric["common_violations"].items():
            # Naive check: if pattern matches and missing missing
            # In real system: Symbolic Algebra Check
            pattern_match = all(term in answer_norm for term in v_data["pattern"])
            missing_check = all(term not in answer_norm for term in v_data.get("missing", []))
            
            if pattern_match and missing_check:
                return {
                    "status": "AXIOM_VIOLATION",
                    "violation_type": violation_id,
                    "axiom_ref": v_data["axiom_ref"],
                    "feedback": v_data["feedback"],
                    "grade": 0.0
                }

        # 2. Check for Correctness (Required Structure)
        reqs = rubric["required_structure"]
        
        # Check terms
        terms_present = [term in answer_norm for term in reqs.get("terms", [])]
        final_state_match = reqs.get("final_state", "") in answer_norm if reqs.get("final_state") else True
        
        if all(terms_present) and final_state_match:
            return {
                "status": "CORRECT",
                "feedback": "Structure Valid. Axioms Respected.",
                "grade": 1.0
            }
            
        # 3. Partial / Incomplete
        missing_terms = [t for t, p in zip(reqs.get("terms", []), terms_present) if not p]
        
        return {
            "status": "INCOMPLETE",
            "missing_terms": missing_terms,
            "feedback": f"Respuesta incompleta. Faltan elementos estructurales: {missing_terms}",
            "grade": 0.5
        }

    def _normalize(self, text: str) -> str:
        # Remove spaces, lower case for simple matching
        # "a^2 + 2ab + b^2" -> "a^2+2ab+b^2"
        # "x = 7" -> "x=7"
        return text.replace(" ", "").lower()
