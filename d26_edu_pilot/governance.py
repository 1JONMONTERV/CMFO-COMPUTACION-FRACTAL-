import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class EducationalResponse:
    content: str
    is_allowed: bool
    status: str
    pedagogical_note: str

class CurriculumCompiler:
    def __init__(self, syllabus_path: str):
        self.syllabus = self._load_syllabus(syllabus_path)
        self.forbidden_keywords = self.syllabus['cmfo_domain']['forbidden_topics']
        
        # Flatten allowed topics for keyword matching
        self.allowed_leaves = []
        self._extract_leaves(self.syllabus['cmfo_domain']['allowed_topics'])
        
        # Load Axiomatic Violations
        self.axioms = self.syllabus['cmfo_domain']['axioms']

    def _load_syllabus(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_leaves(self, node):
        if isinstance(node, list):
            for item in node:
                self._extract_leaves(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                self._extract_leaves(value)
        elif isinstance(node, str):
            self.allowed_leaves.append(node)

class Math10DomainValidator:
    def __init__(self, syllabus_path: str):
        self.compiler = CurriculumCompiler(syllabus_path)
        self.policy = self.compiler.syllabus['cmfo_domain']['response_policy']

    def validate_query(self, query: str) -> EducationalResponse:
        query_lower = query.lower()

        # 1. Check for Forbidden Topics (Hard Block)
        for prohibited in self.compiler.forbidden_keywords:
            # Simple keyword matching for prototype - in prod would use semantic embedding
            if prohibited.replace("_", " ") in query_lower or prohibited in query_lower:
                msg = self.policy['out_of_scope_query']['message']
                return EducationalResponse(
                    content=msg,
                    is_allowed=False,
                    status="BLOCKED_BY_SYLLABUS",
                    pedagogical_note=f"Detected forbidden concept: {prohibited}"
                )

        # 2. Check for Common Axiomatic Violations (The Pedagogical Core)
        # Scan known violation patterns defined in YAML
        for axiom_name, axiom_data in self.compiler.axioms.items():
            for violation in axiom_data.get('common_violations', []):
                # Normalized check (removing spaces for robust matching in this simple version)
                if self._normalize(violation) in self._normalize(query):
                    return EducationalResponse(
                        content=axiom_data['pedagogical_response'],
                        is_allowed=True, # Allowed to process, but it's an error correction
                        status="AXIOM_VIOLATION_DETECTED",
                        pedagogical_note=f"Correcting violation of {axiom_name}"
                    )

        # 3. Check if topic is allowed (Whitelist)
        # This is a loose check for the prototype. If it's not forbidden and looks math-y, we pass it.
        # In a real system, we'd ensure it maps to an allowed_topic.
        return EducationalResponse(
            content="Consulta válida dentro del currículo de 10mo año.",
            is_allowed=True,
            status="VALID_CURRICULUM_QUERY",
            pedagogical_note="Proceed to Step-by-Step Derivation"
        )
    
    def _normalize(self, text: str) -> str:
        return text.replace(" ", "").replace("^", "").replace("*", "").lower()

# CLI Simulator for testing
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

    validator = Math10DomainValidator("d26-edu-pilot/syllabus_math_10.json")
    
    test_cases = [
        "Quiero calcular la derivada de x^2",
        "Explícame la pendiente de una recta",
        "Por qué (a+b)^2 = a^2 + b^2",
        "Cómo resuelvo integrales",
        "Resuelve 2x + 4 = 10"
    ]

    print(f"[*] CMFO-EDU-MATH-10 PILOT: Governance Test\n")
    print(f"Constitution: {validator.compiler.syllabus['cmfo_domain']['name']}")
    print("-" * 60)

    for q in test_cases:
        print(f"Student: '{q}'")
        resp = validator.validate_query(q)
        print(f"CMFO Status: {resp.status}")
        print(f"Response: {resp.content}")
        print(f"Note: {resp.pedagogical_note}")
        print("-" * 60)
