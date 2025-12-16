import sys
import os
import uuid
from typing import Dict, Any, Optional

# Ensure we can import from root
sys.path.append(os.getcwd())

from d26_edu_pilot.validator import EpistemologicalValidator
from d26_edu_pilot.curriculum_compiler import CurriculumCompiler
from cmfo.actions.gate import ActionGate
from pathlib import Path

class PedagogicalTutor:
    """
    The Orchestrator of the Educational Pilot.
    Integrates Governance (Syllabus) and Execution Safety (ActionGate).
    """

    def __init__(self, syllabus_path: str):
        # 1. Initialize Constitution (Validator)
        self.compiler = CurriculumCompiler(Path(syllabus_path))
        self.profile = self.compiler.compile()
        self.validator = EpistemologicalValidator()
        self.validator.configure_curriculum(self.profile)

        # 2. Initialize Executive Power (ActionGate)
        self.gate = ActionGate()

        # 3. Initialize "Knowledge Base" (Simulated Proof Generation)
        # In a full system, this would be D16 reasoning engine.
        self.knowledge_base = {
            "péndulo": {"proof": "physics_laws", "action": "explain_concept"}, # Intentional typo to test normalized matching? No, "pendiente"
            "pendiente": {"proof": "geom_analytic_def_1", "action": "explain_concept"},
            "recta": {"proof": "euclid_axiom_1", "action": "explain_concept"},
            "2x+4=10": {"proof": "linear_alg_step_1", "action": "solve_step"},
        }

    def process_student_query(self, query: str, user_id: str = "student_01") -> Dict[str, Any]:
        """
        Process a query through the Governance Pipeline.
        """
        print(f"\n[Tutor] Processing: '{query}'")

        # Step 1: Constitutional Check (Syllabus)
        # We need to extract the core concept. 
        # For this prototype, we treat the whole query as a bag of concepts or use naive keyword matching.
        # Let's clean the query to find a concept.
        concept = self._extract_concept(query)
        
        validation = self.validator.validate_concept(concept)
        
        if validation["status"] == "REJECTED":
            # Blocked by Constitution
            return {
                "type": "block",
                "message": validation["pedagogical_redirect"],
                "authority": f"SYLLABUS:{self.profile.domain_name}"
            }

        if validation["status"] == "AXIOM_VIOLATION":
            # Pedagogical Correction
            return {
                "type": "correction",
                "message": validation["pedagogical_redirect"],
                "authority": f"AXIOM:{self.profile.domain_name}"
            }

        # Step 2: Proof Check
        # Does the system have a proof/method for this?
        kb_entry = self._lookup_knowledge(concept, query)
        
        proof_ref = None
        action_type = "general_response"
        
        if kb_entry:
            proof_ref = kb_entry["proof"]
            action_type = kb_entry["action"]
        
        # Step 2.5: Handle conversational queries (greetings, general questions)
        # These don't need strict proof but should be allowed
        conversational_keywords = ["hola", "hello", "hi", "gracias", "ayuda", "help", "qué puedes", "what can"]
        is_conversational = any(kw in query.lower() for kw in conversational_keywords)
        
        if is_conversational:
            return {
                "type": "response",
                "content": self._generate_conversational_response(query),
                "receipt": {"action": "conversational", "confidence": 1.0}
            }
        
        # Step 3: For general math questions without specific proof, allow if validated
        # Instead of blocking, provide a generic helpful response
        if validation["status"] == "ALLOWED" and not proof_ref:
            return {
                "type": "response",
                "content": self._generate_content(action_type, query, "general_math_knowledge"),
                "receipt": {"action": action_type, "confidence": 0.8}
            }
        
        # Step 4: Executive Authorization (ActionGate) - only for specific proofs
        if proof_ref:
            receipt = self.gate.attempt_execution(
                action=action_type,
                domain=self.profile.domain_name,
                input_data=query,
                user_auth=True,
                domain_auth=True,
                proof_ref=proof_ref
            )

            if receipt.confidence == 1.0:
                return {
                    "type": "response",
                    "content": self._generate_content(action_type, query, proof_ref),
                    "receipt": receipt.to_json()
                }
            else:
                return {
                    "type": "error",
                    "message": f"No puedo responder esto. {receipt.result}",
                    "receipt": receipt.to_json()
                }
        
        # Fallback: Unknown but not forbidden
        return {
            "type": "response",
            "content": "Entiendo tu pregunta. ¿Podrías ser más específico sobre qué tema de matemáticas de 10º grado te gustaría explorar? Puedo ayudarte con álgebra lineal, geometría euclidiana o aritmética.",
            "receipt": {"action": "clarification", "confidence": 0.5}
        }

    def _extract_concept(self, query: str) -> str:
        # Naive extraction for prototype
        q = query.lower()
        if "derivada" in q: return "derivada"
        if "integral" in q: return "integral"
        if "pendiente" in q: return "pendiente"
        if "recta" in q: return "recta"
        if "2x+4=10" in q.replace(" ", ""): return "2x+4=10"
        return query # Fallback

    def _lookup_knowledge(self, concept: str, query: str) -> Optional[Dict]:
        return self.knowledge_base.get(concept)

    def _generate_content(self, action: str, query: str, proof_ref: str) -> str:
        # Simulated content generator
        if action == "explain_concept":
            if "pendiente" in query.lower():
                return "La pendiente (m) representa el cambio vertical dividido por el cambio horizontal. m = (y2-y1)/(x2-x1)."
            if "ecuación" in query.lower() or "equation" in query.lower():
                return "Una ecuación lineal tiene la forma ax + b = c. Para resolverla, aislamos la variable x usando operaciones inversas."
        if action == "solve_step":
            return "Paso 1: Restar 4 a ambos lados. 2x = 6."
        if action == "general_response":
            return "Puedo ayudarte con temas de matemáticas de 10º grado: álgebra lineal, geometría euclidiana y aritmética básica. ¿Qué te gustaría aprender?"
        return "Entiendo tu pregunta. Estoy aquí para ayudarte con matemáticas de 10º grado."
    
    def _generate_conversational_response(self, query: str) -> str:
        """Handle greetings and general conversational queries"""
        q = query.lower()
        if "hola" in q or "hello" in q or "hi" in q:
            return "¡Hola! Soy tu tutor de matemáticas para 10º grado. Puedo ayudarte con álgebra lineal, geometría euclidiana y aritmética. ¿En qué tema te gustaría trabajar hoy?"
        if "gracias" in q or "thank" in q:
            return "¡De nada! Estoy aquí para ayudarte a aprender matemáticas."
        if "ayuda" in q or "help" in q or "qué puedes" in q or "what can" in q:
            return "Puedo ayudarte con:\n• Ecuaciones lineales\n• Geometría euclidiana\n• Aritmética y álgebra básica\n\nNo puedo enseñar cálculo ni física cuántica (están fuera del programa de 10º grado)."
        return "Estoy aquí para ayudarte con matemáticas de 10º grado. ¿Tienes alguna pregunta específica?"
