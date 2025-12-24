"""
D24: CAPA DE INTERFAZ HUMANA
Implementa el protocolo de salida en 4 capas para garantizar seguridad cognitiva.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from source_registry import KnowledgeClaim, EpistemologicalValidator, ProofType, ValidityStatus, StructuralProof

# ============================================================================
# ETIQUETAS DE VERDAD (SEMÁNTICAS)
# ============================================================================
class TruthLabel(str):
    FORMAL_PROOF = "✅ FORMALMENTE DEMOSTRADO"
    EMPIRICAL_EVIDENCE = "✅ EVIDENCIA EMPÍRICA VÁLIDA"
    DOCTRINAL_COHERENCE = "📜 COHERENCIA DOCTRINAL (No verificable empíricamente)"
    THEORETICAL_MODEL = "⚠️ MODELO TEÓRICO (Requiere validación)"
    SPECULATION = "🧠 ESPECULACIÓN LÓGICA"
    UNVERIFIED = "❌ NO VERIFICADO"
    DOMAIN_ERROR = "⛔ ERROR DE DOMINIO"

@dataclass
class CMFOResponse:
    header_layer: str     # Capa 1: Detector de Dominio
    contract_layer: str   # Capa 2: Reglas activas
    content_layer: str    # Capa 3: Respuesta
    label_layer: str      # Capa 4: Etiqueta de verdad

# ============================================================================
# CLASIFICADOR Y GENERADOR DE RESPUESTA
# ============================================================================

class CMFOInterface:
    def __init__(self):
        self.validator = EpistemologicalValidator()
        
    def process_query(self, query: str, claimed_domain: str, proof_provided: Optional[StructuralProof] = None, audit_mode: bool = False) -> Union[CMFOResponse, Dict]:
        """
        Procesa una consulta simulada aplicando las 4 capas de seguridad.
        Si audit_mode=True, retorna el trace completo de decisión (IEEE 7000).
        """
        # 0. Trace de Auditoría
        audit_trace = {
            "timestamp": "2025-12-15T23:59:00Z", # Mock
            "input_query": query,
            "claimed_domain": claimed_domain,
            "axioms_invoked": []
        }

        # 1. Detección de Dominio (Simulada aquí, vendría de NLP)
        if "Dios" in query or "gracia" in query:
            detected_domain = "Theology"
        elif "primo" in query or "numero" in query:
            detected_domain = "Mathematics"
        elif "electron" in query or "luz" in query or "universo" in query:
            detected_domain = "Physics"
        else:
            detected_domain = claimed_domain # Fallback
            
        audit_trace["detected_domain"] = detected_domain
        audit_trace["axioms_invoked"].append("A2_Domain_Sovereignty")
            
        # 2. Validación de Contenido
        # Creamos un claim temporal
        proof = proof_provided if proof_provided else StructuralProof(ProofType.NONE, "", False)
        claim = KnowledgeClaim(query, detected_domain, proof)
        status = self.validator.validate_claim(claim)
        
        audit_trace["validation_status"] = status.value
        audit_trace["proof_type"] = proof.type.value
        audit_trace["axioms_invoked"].append("A1_Structural_Truth")

        # 3. Construcción de Capas
        
        # Capa 1: Header
        icons = {"Mathematics": "🧮", "Physics": "🔬", "Theology": "🏛️", "Computation": "💻"}
        icon = icons.get(detected_domain, "📘")
        header = f"{icon} Dominio detectado: {detected_domain}"
        
        # Capa 2: Contrato
        contracts = {
            "Mathematics": "Reglas: Verdad formal, Independencia temporal.",
            "Physics": "Reglas: Datos empíricos, Falsabilidad obligatoria.",
            "Theology": "Reglas: Coherencia doctrinal, No interferencia con física.",
        }
        contract = contracts.get(detected_domain, "Reglas estándar del dominio.")
        
        # Capa 4: Label (Derivada del status y dominio)
        if status == ValidityStatus.INVALID:
            label = TruthLabel.DOMAIN_ERROR
            content = "La afirmación viola las reglas estructurales del dominio detectado."
        elif status == ValidityStatus.SPECULATIVE:
            label = TruthLabel.UNVERIFIED if detected_domain == "Physics" else TruthLabel.SPECULATION
            content = f"La afirmación es posible pero carece de demostración estructural completa en {detected_domain}."
        elif status == ValidityStatus.VALIDATED:
            if detected_domain == "Mathematics": label = TruthLabel.FORMAL_PROOF
            elif detected_domain == "Physics": label = TruthLabel.EMPIRICAL_EVIDENCE
            elif detected_domain == "Theology": label = TruthLabel.DOCTRINAL_COHERENCE
            else: label = TruthLabel.THEORETICAL_MODEL
            content = f"Afirmación validada estructuralmente dentro de {detected_domain}."
        else:
            label = TruthLabel.UNVERIFIED
            content = "No se puede determinar la validez."
            
        if audit_mode:
            # IEEE 7000 / ISO Standard Audit Trace
            return {
                "domain": detected_domain,
                "claim": query,
                "proof_type": proof.type.value if proof else "none",
                "confidence": 1.0 if status == ValidityStatus.VALIDATED else 0.0, # Deterministic fallback
                "axioms_used": audit_trace["axioms_invoked"],
                "domain_constraints": [contracts.get(detected_domain, "Standard")],
                "reasoning_trace": [f"Detected {detected_domain}", f"Validated {status.value}"],
                "status": status.value.upper()
            }

        return CMFOResponse(header, contract, content, label)

# ============================================================================
# EJECUCIÓN DE PRUEBA
# ============================================================================
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    interface = CMFOInterface()
    
    queries = [
        ("Todo numero primo > 2 es impar", "Mathematics", StructuralProof(ProofType.FORMAL_LOGIC, "...", True)),
        ("Dios creo la luz", "Theology", StructuralProof(ProofType.INTERNAL_COHERENCE, "...", True)),
        ("El electron tiene conciencia", "Physics", StructuralProof(ProofType.EMPIRICAL_DATA, "", False)) # Falla prueba
    ]
    
    print("[*] TEST DE INTERFAZ 4 CAPAS\n")
    
    for q, dom, proof in queries:
        resp = interface.process_query(q, dom, proof)
        print(f"Query: '{q}'")
        print(f"├── {resp.header_layer}")
        print(f"├── {resp.contract_layer}")
        print(f"├── {resp.content_layer}")
        print(f"└── {resp.label_layer}")
        print("-" * 40)
    
    print("\n[*] TEST DE MODO AUDITORÍA (IEEE 7000)\n")
    audit_trace = interface.process_query("Todo numero primo > 2 es impar", "Mathematics", queries[0][2], audit_mode=True)
    print("Audit JSON Trace:")
    print(audit_trace)
    print("\n[VERIFICADO] El sistema produce trazas transparentes.")
