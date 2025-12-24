"""
D23: GENERADOR DE EXIGENCIAS FORMALES
El sistema legislativo que emite restricciones y obligaciones ontológicas.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

# ============================================================================
# 1. ESTRUCTURAS DE DATOS
# ============================================================================

class ExigenceType(Enum):
    CONSTRAINT = "constraint"  # Restricción dura (debe cumplirse)
    ACTION = "action"          # Acción requerida (debe ejecutarse)
    WARNING = "warning"        # Advertencia ontológica (riesgo)

@dataclass
class ProofReference:
    """Referencia al origen de la exigencia"""
    source_axiom: str
    domain: str
    authority_level: float = 1.0  # 1.0 = Axioma soberano

@dataclass
class DomainExigence:
    """Una exigencia formal emitida por un dominio"""
    domain: str
    type: ExigenceType
    description: str
    justification: ProofReference
    priority: float
    target_context: Optional[str] = None

# ============================================================================
# 2. GENERADORES DE EXIGENCIAS POR DOMINIO
# ============================================================================

class ExigenceGenerator:
    """Clase base para generadores de exigencias"""
    
    def generate(self, context: Dict) -> List[DomainExigence]:
        """Genera exigencias basadas en el contexto dado"""
        raise NotImplementedError

class MathExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        exigences = []
        
        # Constraint: Verdad no depende del tiempo
        exigences.append(DomainExigence(
            domain="Mathematics",
            type=ExigenceType.CONSTRAINT,
            description="No introducir dependencias temporales en pruebas",
            justification=ProofReference("Tiempo = 0", "Mathematics"),
            priority=1.0
        ))
        
        # Action: Permitir infinito
        if context.get("involves_infinity", False):
            exigences.append(DomainExigence(
                domain="Mathematics",
                type=ExigenceType.ACTION,
                description="Tratar infinito como objeto simbolico valido",
                justification=ProofReference("Existencia simbolica permitida", "Mathematics"),
                priority=0.9
            ))
            
        return exigences

class CompExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        exigences = []
        
        # Constraint: Terminación
        exigences.append(DomainExigence(
            domain="Computation",
            type=ExigenceType.CONSTRAINT,
            description="Todo proceso debe garantizar terminacion o declararse indefinido",
            justification=ProofReference("Recursos finitos", "Computation"),
            priority=1.0
        ))
        
        # Warning: Infinito
        if context.get("involves_infinity", False):
            exigences.append(DomainExigence(
                domain="Computation",
                type=ExigenceType.WARNING,
                description="Infinito detectado: requiere aproximacion o lazy evaluation",
                justification=ProofReference("Finitud de memoria", "Computation"),
                priority=0.8
            ))
            
        return exigences

class QuantumExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        return [
            DomainExigence(
                domain="QuantumComp",
                type=ExigenceType.CONSTRAINT,
                description="Prohibido clonar estados cuanticos arbitrarios",
                justification=ProofReference("Teorema de No-Clonacion", "QuantumComp"),
                priority=1.0
            ),
            DomainExigence(
                domain="QuantumComp",
                type=ExigenceType.ACTION,
                description="Formular operaciones como transformaciones unitarias",
                justification=ProofReference("Reversibilidad cuantica", "QuantumComp"),
                priority=0.9
            )
        ]

class PhysicsExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        return [
            DomainExigence(
                domain="Physics",
                type=ExigenceType.CONSTRAINT,
                description="Verdad requiere contraste experimental o consistencia teorica",
                justification=ProofReference("Empirismo", "Physics"),
                priority=1.0
            ),
            DomainExigence(
                domain="Physics",
                type=ExigenceType.WARNING,
                description="No proyectar axiomas matematicos como realidades fisicas sin prueba",
                justification=ProofReference("Mapa != Territorio", "Physics"),
                priority=0.8
            )
        ]

class BiologyExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        return [
            DomainExigence(
                domain="Biology",
                type=ExigenceType.CONSTRAINT,
                description="No reducir sistemas complejos a fisica determinista simple",
                justification=ProofReference("Autopoiesis", "Biology"),
                priority=0.9
            )
        ]

class MedicineExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        return [
            DomainExigence(
                domain="Medicine",
                type=ExigenceType.CONSTRAINT,
                description="Primacia de la seguridad del paciente (Primum Non Nocere)",
                justification=ProofReference("Etica Medica", "Medicine"),
                priority=1.0
            ),
            DomainExigence(
                domain="Medicine",
                type=ExigenceType.WARNING,
                description="Certeza absoluta imposible; gestionar riesgo probabilisticamente",
                justification=ProofReference("Variabilidad Biologica", "Medicine"),
                priority=0.9
            )
        ]

class SocialSciExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        return [
            DomainExigence(
                domain="SocialSci",
                type=ExigenceType.WARNING,
                description="Considerar contexto cultural y normativo",
                justification=ProofReference("No universalidad social", "SocialSci"),
                priority=0.8
            )
        ]

class LanguageExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        return [
             DomainExigence(
                domain="Language",
                type=ExigenceType.WARNING,
                description="El lenguaje no garantiza verdad ontologica",
                justification=ProofReference("Mapa != Territorio", "Language"),
                priority=0.7
            )
        ]

class EthicsExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        return [
            DomainExigence(
                domain="Ethics",
                type=ExigenceType.CONSTRAINT,
                description="Declaraciones normativas (debe ser) no son facticas (es)",
                justification=ProofReference("Guillotina de Hume", "Ethics"),
                priority=1.0
            )
        ]

class MetaphysicsExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        exigences = []
        
        # Constraint: No falsable
        exigences.append(DomainExigence(
            domain="Metaphysics",
            type=ExigenceType.WARNING,
            description="Afirmaciones no son falsables empiricamente",
            justification=ProofReference("Naturaleza abstracta", "Metaphysics"),
            priority=0.9
        ))
        
        # Constraint: Coherencia interna
        exigences.append(DomainExigence(
            domain="Metaphysics",
            type=ExigenceType.CONSTRAINT,
            description="Coherencia logica interna obligatoria",
            justification=ProofReference("Principio de No Contradiccion", "Metaphysics"),
            priority=1.0
        ))
        
        # Non-contamination constraint
        exigences.append(DomainExigence(
            domain="Metaphysics",
            type=ExigenceType.CONSTRAINT,
            description="NO proyectar a Fisica o Computacion sin puente explicito",
            justification=ProofReference("Regla de No-Contaminacion", "CMFO"),
            priority=1.0
        ))
            
        return exigences

class TheologyExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        exigences = []
        
        # Sovereign Axiom: Revelación
        exigences.append(DomainExigence(
            domain="Theology",
            type=ExigenceType.ACTION,
            description="Evaluar coherencia doctrinal interna",
            justification=ProofReference("Sistema Doctrinal", "Theology"),
            priority=0.9
        ))
        
        # Non-contamination constraint
        exigences.append(DomainExigence(
            domain="Theology",
            type=ExigenceType.CONSTRAINT,
            description="PROHIBIDO proyectar dogmas como hechos fisicos o matematicos",
            justification=ProofReference("Regla de No-Contaminacion", "CMFO"),
            priority=1.0
        ))
        
        return exigences

class CMFOExigenceGenerator(ExigenceGenerator):
    def generate(self, context: Dict) -> List[DomainExigence]:
        return [
            DomainExigence(
                domain="CMFO",
                type=ExigenceType.ACTION,
                description="Mapear tensiones interdominio y preservar coherencia fractal",
                justification=ProofReference("Axioma Fractal", "CMFO"),
                priority=1.0
            ),
             DomainExigence(
                domain="CMFO",
                type=ExigenceType.CONSTRAINT,
                description="Arbitrar conflictos usando axioma de coexistencia",
                justification=ProofReference("Arbitraje Ontologico", "CMFO"),
                priority=1.0
            )
        ]


# ============================================================================
# 3. GESTOR CENTRAL
# ============================================================================

class ExigenceManager:
    """Orquesta la generacion y auditoria de exigencias"""
    
    def __init__(self):
        self.generators = {
            "Mathematics": MathExigenceGenerator(),
            "Computation": CompExigenceGenerator(),
            "QuantumComp": QuantumExigenceGenerator(),
            "Physics": PhysicsExigenceGenerator(),
            "Biology": BiologyExigenceGenerator(),
            "Medicine": MedicineExigenceGenerator(),
            "SocialSci": SocialSciExigenceGenerator(),
            "Language": LanguageExigenceGenerator(),
            "Ethics": EthicsExigenceGenerator(),
            "Metaphysics": MetaphysicsExigenceGenerator(),
            "Theology": TheologyExigenceGenerator(),
            "CMFO": CMFOExigenceGenerator()
        }
        self.auditor = ExigenceAuditor()
        
    def get_exigences_for_domains(self, domains: List[str], context: Dict) -> Dict[str, List[DomainExigence]]:
        """Obtiene y audita exigencias para una lista de dominios activos"""
        results = {}
        all_exigences = []
        
        for domain in domains:
            if domain in self.generators:
                raw_exigences = self.generators[domain].generate(context)
                
                # Auditar exigencias
                valid_exigences = []
                for ex in raw_exigences:
                    if self.auditor.validate(ex):
                        valid_exigences.append(ex)
                        all_exigences.append(ex)
                        
                results[domain] = valid_exigences
                
        # Verificar conflictos entre exigencias de distintos dominios
        conflicts = self.auditor.detect_conflicts(all_exigences)
        if conflicts:
            results["_CONFLICTS"] = conflicts
            
        return results

class ExigenceAuditor:
    """Valida exigencias y asegura no-contaminacion"""
    
    def validate(self, exigence: DomainExigence) -> bool:
        """
        Valida una exigencia individual.
        Evita 'alucinacion de reglas'.
        """
        # Placeholder logic: validation is implicit in the hardcoded generators for now.
        # In a dynamic system, this would check against a knowledge base of axioms.
        return True

    def detect_conflicts(self, exigences: List[DomainExigence]) -> List[str]:
        """Detecta tensiones entre grupos de exigencias"""
        conflicts = []
        
        # Ejemplo: Infinidad Matematica vs Finitud Computacional
        has_infinity_allowed = any(e.domain == "Mathematics" and "infinito" in e.description for e in exigences)
        has_infinity_forbidden = any(e.domain == "Computation" and "terminacion" in e.description for e in exigences)
        
        if has_infinity_allowed and has_infinity_forbidden:
            conflicts.append("TENSION DETECTADA: Infinito Matematico vs Finitud Computacional")
            
        # Ejemplo: Verdad Empirica vs Verdad Revelada
        has_empirical = any(e.domain == "Physics" and "experimental" in e.description for e in exigences)
        has_revelation = any(e.domain == "Theology" and "doctrinal" in e.description for e in exigences if e.type == ExigenceType.ACTION)
        
        # Nota: La teologia tiene prohibido hacer claims fisicos, asi que esto es solo si ambas estan activas
        if has_empirical and has_revelation:
             # Esto no es necesariamente un conflicto si estan en dominios separados, 
             # pero D22 debe arbitrar.
             pass 
             
        return conflicts

# ============================================================================
# 4. EJECUCION DE PRUEBA
# ============================================================================

if __name__ == "__main__":
    manager = ExigenceManager()
    
    print("[*] INICIANDO D23: SISTEMA DE EXIGENCIAS FORMALES")
    print("-" * 50)
    
    # Escenario 1: Analisis de Infinito (Math + Comp)
    print("\n[ESCENARIO 1] Concepto: Serie Infinita")
    ctx1 = {"involves_infinity": True}
    res1 = manager.get_exigences_for_domains(["Mathematics", "Computation"], ctx1)
    
    for dom, ex_list in res1.items():
        if dom == "_CONFLICTS":
            for c in ex_list: print(f"  [!] {c}")
        else:
            for ex in ex_list:
                print(f"  [{dom}] {ex.type.value.upper()}: {ex.description}")
                
    # Escenario 2: Metafisica y Teologia (Non-contamination check)
    print("\n[ESCENARIO 2] Concepto: Causa Primera / Dios")
    ctx2 = {}
    res2 = manager.get_exigences_for_domains(["Metaphysics", "Theology", "Physics"], ctx2)
    
    for dom, ex_list in res2.items():
         if dom != "_CONFLICTS":
            for ex in ex_list:
                # Filtrar solo restricciones de no-contaminación para la demo
                if "PROHIBIDO" in ex.description or "NO proyectar" in ex.description or "contaminacion" in ex.description.lower():
                     print(f"  [{dom}] {ex.type.value.upper()}: {ex.description}")

    print("-" * 50)
    print("[OK] D23 OPERATIVO: Legislacion Ontologica Activa")
