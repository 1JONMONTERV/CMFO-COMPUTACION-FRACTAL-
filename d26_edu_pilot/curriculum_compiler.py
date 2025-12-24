import json
from pathlib import Path
from typing import Set, Dict, Any
from .curricular_profile import CurricularProfile

class CurriculumCompiler:
    """
    Compila un syllabus JSON en un perfil epistemológico ejecutable.
    """

    def __init__(self, syllabus_path: Path):
        self.syllabus_path = syllabus_path

    def compile(self) -> CurricularProfile:
        with open(self.syllabus_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        domain = raw["cmfo_domain"]

        allowed = self._flatten_topics(domain["allowed_topics"])
        forbidden = set(domain["forbidden_topics"])

        # Extract specific redirections from response policy if structured, 
        # or simplified default map for now based on the prompt design
        redirections = {} 
        # In the prompt, redirections were shown as coming from "response_policy.out_of_scope_query"
        # Since the JSON has a single message, we will map forbidden topics to that generic message for now,
        # or distinct ones if the JSON had them. The prompt implies a dict map.
        # Let's check the JSON structure provided in Step 1539.
        # It has "out_of_scope_query": { "message": "..." }
        # We will store the generic message.
        self.generic_redirect = domain.get("response_policy", {}).get("out_of_scope_query", {}).get("message", "")

        axioms = domain.get("axioms", {})
        pedagogy = domain.get("response_policy", {})

        return CurricularProfile(
            domain_name=domain["name"],
            grade_level=domain["grade_level"],
            allowed_topics=allowed,
            forbidden_topics=forbidden,
            redirections={"*": self.generic_redirect}, # Using a wildcard for the generic message
            axioms=axioms,
            pedagogy_policy=pedagogy
        )

    def _flatten_topics(self, topics_dict) -> Set[str]:
        """
        Convierte estructura jerárquica en conjunto plano:
        algebra.polynomials.square_of_a_binomial -> string único
        """
        flat = set()

        def walk(prefix, obj):
            if isinstance(obj, list):
                for x in obj:
                    if isinstance(x, dict):
                        # Handle case like {"special_products": [...]} inside a list
                        for k, v in x.items():
                             walk(f"{prefix}.{k}" if prefix else k, v)
                    else:
                        flat.add(f"{prefix}.{x}" if prefix else x)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}.{k}" if prefix else k, v)
            else:
                flat.add(f"{prefix}.{obj}" if prefix else obj)

        walk("", topics_dict)
        return flat
