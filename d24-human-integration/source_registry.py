"""
D24: REGISTRO DE FUENTES Y VALIDACIÓN EPISTEMOLÓGICA
Implementa el Axioma CMFO de Verdad Independiente del Autor.
Verdad := Coherencia Estructural + Validez Demostrativa en su Dominio.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import datetime

# ============================================================================
# 1. DEFINICIONES DE ENTRADA Y VERDAD
# ============================================================================

class ProofType(Enum):
    FORMAL_LOGIC = "formal_logic"             # Matemáticas / Lógica
    EMPIRICAL_DATA = "empirical_data"         # Física / Biología
    ALGORITHMIC_COMPLEXITY = "algorithmic"    # Computación
    INTERNAL_COHERENCE = "internal_coherence" # Teología / Metafísica
    NORMATIVE_REFERENCE = "normative"         # Ética / Derecho
    NONE = "none"

class ValidityStatus(Enum):
    VALIDATED = "validated"          # Demostración estructural correcta
    SPECULATIVE = "speculative"      # Posible pero no demostrado
    INVALID = "invalid"              # Falla estructural o contradicción
    UNKNOWN = "unknown"              # Falta información

@dataclass
class AuthorIdentity:
    """Metadato de trazabilidad, NO de validación"""
    name: str
    is_verified: bool = False
    domains: List[str] = field(default_factory=list)

@dataclass
class StructuralProof:
    """La unidad real de validación"""
    type: ProofType
    content: str  # La demostración en sí (o referencia a ella)
    is_structurally_sound: bool = False

@dataclass
class KnowledgeClaim:
    """
    La unidad atómica de conocimiento.
    Fuente Real := (Dominio, Estructura, Demostración)
    """
    content: str
    domain: str
    proof: StructuralProof
    author: Optional[AuthorIdentity] = None  # Opcional, metadato
    venue: Optional[str] = None              # Irrelevante para validación (ej: "Nature", "Blog")
    
    @property
    def has_valid_proof(self) -> bool:
        return self.proof.is_structurally_sound

# ============================================================================
# 2. VALIDACIÓN EPISTEMOLÓGICA
# ============================================================================

class EpistemologicalValidator:
    """
    El núcleo que decide la verdad basándose en estructura, ignorando prestigio.
    """
    
    def validate_claim(self, claim: KnowledgeClaim) -> ValidityStatus:
        # Regla 1: Chequeo de Dominio vs Tipo de Prueba
        if not self._check_domain_proof_match(claim.domain, claim.proof.type):
            return ValidityStatus.INVALID
            
        # Regla 2: Validez Estructural de la Demostración
        # (Aquí CMFO verificaría la lógica interna. Mocked por 'is_structurally_sound')
        if claim.proof.is_structurally_sound:
            return ValidityStatus.VALIDATED
            
        # Regla 3: Si no hay demostración sólida, es especulación
        return ValidityStatus.SPECULATIVE

    def _check_domain_proof_match(self, domain: str, proof_type: ProofType) -> bool:
        """Asegura que no se use teología para probar física, etc."""
        mapping = {
            "Mathematics": [ProofType.FORMAL_LOGIC],
            "Physics": [ProofType.EMPIRICAL_DATA, ProofType.FORMAL_LOGIC], # Física teórica usa lógica
            "Computation": [ProofType.ALGORITHMIC_COMPLEXITY, ProofType.FORMAL_LOGIC],
            "Theology": [ProofType.INTERNAL_COHERENCE],
            "Metaphysics": [ProofType.INTERNAL_COHERENCE, ProofType.FORMAL_LOGIC],
            "Ethics": [ProofType.NORMATIVE_REFERENCE]
        }
        # Normalización simple de strings
        for d_key, p_types in mapping.items():
            if d_key.lower() in domain.lower():
                return proof_type in p_types
        return True # Default permisivo para otros dominios

# ============================================================================
# 3. REGISTRO DE CONOCIMIENTO (NO DE AUTORES)
# ============================================================================

class SourceRegistry:
    def __init__(self):
        self.validator = EpistemologicalValidator()
        self.knowledge_base: List[Dict] = []
    
    def register_claim(self, claim: KnowledgeClaim) -> Dict:
        """
        Ingesta una afirmación, la valida y emite un veredicto.
        Ignora totalmente si el autor es famoso o anónimo.
        """
        status = self.validator.validate_claim(claim)
        
        record = {
            "content": claim.content,
            "status": status.value,
            "proof_type": claim.proof.type.value,
            "has_author": claim.author is not None,
            "author_name": claim.author.name if claim.author else "ANONYMOUS",
            "venue_ignored": claim.venue # Guardado pero no usado para status
        }
        
        self.knowledge_base.append(record)
        return record

# ============================================================================
# 4. SIMULACIÓN DE CASOS REALES
# ============================================================================

if __name__ == "__main__":
    registry = SourceRegistry()
    
    print("[*] INICIANDO D24: VALIDACIÓN EPISTEMOLÓGICA (PRESTIGIO IGNORADO)")
    print("-" * 60)
    
    # CASO 1: Anónimo con Demostración Matemática Sólida
    print("\n[CASO 1] Anónimo demuestra Teorema")
    c1 = KnowledgeClaim(
        content="Todo número primo > 2 es impar",
        domain="Mathematics",
        proof=StructuralProof(ProofType.FORMAL_LOGIC, "Demostración por reducción...", True),
        author=None, # Nadie sabe quién fue
        venue="Servilleta de bar"
    )
    r1 = registry.register_claim(c1)
    print(f"  Entrada: '{c1.content}' en '{c1.venue}'")
    print(f"  Resultado: {r1['status'].upper()} (Proof: {r1['proof_type']})")
    
    # CASO 2: Físico Famoso sin Datos Empíricos para afirmación Física
    print("\n[CASO 2] Premio Nobel afirma especulación sin datos")
    c2 = KnowledgeClaim(
        content="Existen universos paralelos indetectables",
        domain="Physics",
        proof=StructuralProof(ProofType.EMPIRICAL_DATA, "Intuición experta", False), # Falla estructural
        author=AuthorIdentity("Dr. Famoso", True),
        venue="Nature Cover Story"
    )
    r2 = registry.register_claim(c2)
    print(f"  Entrada: '{c2.content}' por {c2.author.name} en {c2.venue}")
    print(f"  Resultado: {r2['status'].upper()} (Proof: {r2['proof_type']})")
    
    # CASO 3: Teología correcta en su dominio
    print("\n[CASO 3] Afirmación Teológica Coherente")
    c3 = KnowledgeClaim(
        content="La gracia perfecciona la naturaleza",
        domain="Theology",
        proof=StructuralProof(ProofType.INTERNAL_COHERENCE, "Suma Teológica", True),
        author=AuthorIdentity("Tomas de Aquino", True),
        venue="Libro Antiguo"
    )
    r3 = registry.register_claim(c3)
    print(f"  Entrada: '{c3.content}'")
    print(f"  Resultado: {r3['status'].upper()} (Proof: {r3['proof_type']})")

    # CASO 4: Contaminación (Teología intentando probar Física)
    print("\n[CASO 4] Intento de Contaminación (Teología -> Física)")
    c4 = KnowledgeClaim(
        content="La luz viaja instantáneamente por voluntad divina",
        domain="Physics", # Dominio incorrecto para el tipo de prueba
        proof=StructuralProof(ProofType.INTERNAL_COHERENCE, "Argumento doctrinal", True),
        author=AuthorIdentity("Teólogo X", True),
        venue="Blog"
    )
    r4 = registry.register_claim(c4)
    print(f"  Entrada: '{c4.content}'")
    print(f"  Resultado: {r4['status'].upper()} (Mismatch detectado)")

    print("-" * 60)
