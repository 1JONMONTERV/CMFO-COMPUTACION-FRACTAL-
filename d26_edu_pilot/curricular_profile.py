from dataclasses import dataclass
from typing import Set, Dict, Any

@dataclass(frozen=True)
class CurricularProfile:
    """
    Representación inmutable de un perfil curricular compilado.
    Actúa como la "Constitución" en tiempo de ejecución.
    """
    domain_name: str
    grade_level: int

    allowed_topics: Set[str]
    forbidden_topics: Set[str]

    redirections: Dict[str, str]
    axioms: Dict[str, dict]

    pedagogy_policy: Dict[str, Any]
