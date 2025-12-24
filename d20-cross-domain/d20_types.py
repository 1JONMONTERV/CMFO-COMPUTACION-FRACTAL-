from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any

class Domain(Enum):
    MATHEMATICS_INFINITE = "math_inf"
    MATHEMATICS_FINITE = "math_fin"
    PHYSICS_CLASSICAL = "phys_class"
    PHYSICS_QUANTUM = "phys_quant"
    COMPUTATION_DETERMINISTIC = "comp_det"
    COMPUTATION_QUANTUM = "comp_quant"
    CMFO_7D = "cmfo_7d"
    PHILOSOPHY = "phil"
    ENGINEERING = "eng"

@dataclass
class Concept:
    """Representación canónica de un concepto"""
    identifier: str
    domain: Domain
    properties: Dict[str, Any]
    formal_definition: str
    constraints: List[str]
    
@dataclass
class TranslationResult:
    """Resultado de una traducción entre dominios"""
    concept: Concept
    metrics: Dict[str, float]
    warnings: List[str]
    transformation_trace: List[str]
    confidence: float
