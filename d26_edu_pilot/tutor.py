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
            "pendiente": {"proof": "geom_analytic_def_1", "action": "explain_concept"},
            "recta": {"proof": "euclid_axiom_1", "action": "explain_concept"},
            "ecuación": {"proof": "algebra_linear_1", "action": "explain_concept"},
            "ecuacion": {"proof": "algebra_linear_1", "action": "explain_concept"},
            "equation": {"proof": "algebra_linear_1", "action": "explain_concept"},
            "lineal": {"proof": "algebra_linear_1", "action": "explain_concept"},
            "linear": {"proof": "algebra_linear_1", "action": "explain_concept"},
            "resolver": {"proof": "algebra_solve_1", "action": "solve_step"},
            "solve": {"proof": "algebra_solve_1", "action": "solve_step"},
            "resuelve": {"proof": "algebra_solve_1", "action": "solve_step"},
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
        # Improved extraction for prototype
        q = query.lower()
        
        # Forbidden topics
        if "derivada" in q: return "derivada"
        if "integral" in q: return "integral"
        if "cálculo" in q or "calculus" in q: return "derivada"
        
        # Allowed topics - check for keywords
        if "ecuación" in q or "ecuacion" in q or "equation" in q:
            return "ecuación"
        if "lineal" in q or "linear" in q:
            return "lineal"
        if "resolver" in q or "resuelve" in q or "solve" in q:
            return "resolver"
        if "pendiente" in q: 
            return "pendiente"
        if "recta" in q: 
            return "recta"
        
        # Check for equation patterns (contains x and =)
        if "x" in q and "=" in q:
            return "resolver"
            
        return query # Fallback

    def _lookup_knowledge(self, concept: str, query: str) -> Optional[Dict]:
        return self.knowledge_base.get(concept)

    def _generate_content(self, action: str, query: str, proof_ref: str) -> str:
        # Improved content generator
        q = query.lower()
        
        if action == "explain_concept":
            if "pendiente" in q:
                return "La pendiente (m) representa el cambio vertical dividido por el cambio horizontal.\n\nFórmula: m = (y₂-y₁)/(x₂-x₁)\n\nEjemplo: Si tenemos los puntos (1,2) y (3,6):\nm = (6-2)/(3-1) = 4/2 = 2"
            
            if "ecuación" in q or "ecuacion" in q or "equation" in q:
                return "Una ecuación lineal tiene la forma: ax + b = c\n\nPara resolverla:\n1. Aísla los términos con x en un lado\n2. Aísla los términos constantes en el otro lado\n3. Despeja x dividiendo\n\nEjemplo: 2x + 3 = 7\n→ 2x = 7 - 3\n→ 2x = 4\n→ x = 2"
            
            if "lineal" in q or "linear" in q:
                return "El álgebra lineal estudia ecuaciones de primer grado (sin exponentes mayores a 1).\n\nFormas comunes:\n• ax + b = c (ecuación simple)\n• y = mx + b (forma pendiente-intersección)\n• ax + by = c (forma general)\n\n¿Qué tipo de ecuación te gustaría explorar?"
                
        if action == "solve_step" or "resolver" in q or "resuelve" in q or "solve" in q:
            # Try to solve the equation if present
            solution = self._solve_linear_equation(query)
            if solution:
                return solution
            return "Para resolver ecuaciones, necesito ver la ecuación en formato claro.\nEjemplo: 2x + 3 = 7 o 5x - 2 = 3x + 4"
            
        if action == "general_response":
            return "Puedo ayudarte con:\n• Ecuaciones lineales (resolver y explicar)\n• Geometría analítica (pendiente, rectas)\n• Álgebra básica\n\n¿Qué tema específico te interesa?"
            
        return "Entiendo tu pregunta sobre matemáticas de 10º grado. ¿Podrías reformularla o ser más específico?"
    
    
    def _solve_linear_equation(self, query: str) -> str:
        """
        Solve linear equation using 100% CMFO algebra (no regex).
        
        Uses cmfo.education.equation_solver for pure geometric parsing.
        """
        try:
            from cmfo.education.equation_solver import solve_equation_cmfo
            solution = solve_equation_cmfo(query)
            return solution if solution else ""
        except Exception as e:
            # Fallback to simple response
            return f"Error al resolver: {str(e)}"

            
        # Split by =
        parts = clean.split("=")
        if len(parts) != 2:
            return None
            
        left, right = parts[0], parts[1]
        
        try:
            # Simple parser for linear equations
            # Extract coefficients for x and constants
            def parse_side(expr):
                """Parse one side of equation, return (x_coef, constant)"""
                x_coef = 0
                constant = 0
                
                # Add + at start if doesn't start with - or +
                if expr and expr[0] not in ['+', '-']:
                    expr = '+' + expr
                
                # Find all terms
                terms = re.findall(r'[+-][^+-]+', expr)
                
                for term in terms:
                    term = term.strip()
                    if 'x' in term:
                        # Extract coefficient
                        coef_str = term.replace('x', '').strip()
                        if coef_str in ['+', '']:
                            x_coef += 1
                        elif coef_str == '-':
                            x_coef -= 1
                        else:
                            x_coef += float(coef_str)
                    else:
                        # It's a constant
                        if term.strip():
                            constant += float(term)
                
                return x_coef, constant
            
            left_x, left_c = parse_side(left)
            right_x, right_c = parse_side(right)
            
            # Move all x to left, constants to right
            # left_x*x + left_c = right_x*x + right_c
            # (left_x - right_x)*x = right_c - left_c
            
            final_x_coef = left_x - right_x
            final_constant = right_c - left_c
            
            if abs(final_x_coef) < 0.0001:
                return "Esta ecuación no tiene solución única (los coeficientes de x se cancelan)."
            
            solution = final_constant / final_x_coef
            
            # Format solution with steps
            result = "**Resolución paso a paso:**\n\n"
            result += f"Ecuación original:\n{left} = {right}\n\n"
            
            if right_x != 0:
                result += f"Paso 1: Mover términos con x al lado izquierdo\n"
                if right_x > 0:
                    result += f"{left} - {right_x}x = {right_c}\n"
                else:
                    result += f"{left} + {abs(right_x)}x = {right_c}\n"
                result += f"→ {final_x_coef}x + {left_c} = {right_c}\n\n"
            
            if left_c != 0:
                result += f"Paso 2: Mover constantes al lado derecho\n"
                if left_c > 0:
                    result += f"{final_x_coef}x = {right_c} - {left_c}\n"
                else:
                    result += f"{final_x_coef}x = {right_c} + {abs(left_c)}\n"
                result += f"→ {final_x_coef}x = {final_constant}\n\n"
            
            result += f"Paso 3: Despejar x dividiendo\n"
            result += f"x = {final_constant} / {final_x_coef}\n"
            result += f"\n**x = {solution}**"
            
            # Verification
            result += f"\n\nVerificación:\n"
            left_check = left_x * solution + left_c
            right_check = right_x * solution + right_c
            result += f"Lado izquierdo: {left_x}({solution}) + {left_c} = {left_check:.2f}\n"
            result += f"Lado derecho: {right_x}({solution}) + {right_c} = {right_check:.2f}\n"
            result += "✓ Correcto" if abs(left_check - right_check) < 0.01 else "✗ Error"
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Could not parse equation: {e}")
            return None
    
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
