"""
CMFO D8-CORE: Semantic Algebra
==============================
Real Semantic Calculation Engine.

Calculates the 'real value' (7D vector) of words based on algebraic definitions.
Replaces blind hashing with ontological derivation.

AXIOMS (7D Basis):
0. EXISTENCE (Subject/Object)
1. TRUTH (True/False)
2. ORDER (Order/Chaos)
3. ACTION (Dynamic/Static)
4. CONNECTION (Union/Isolation)
5. MIND (Conscious/Inert)
6. TIME (Future/Past)
"""

import math
from typing import List, Dict, Optional

# 1. AXIOMS (Immutable Basis)
AXES = [
    "existence",  # 0
    "truth",      # 1
    "order",      # 2
    "action",     # 3
    "connection", # 4
    "mind",       # 5
    "time",       # 6
]

DIM = 7

# 2. ONTOLOGICAL PROPERTIES (Generators)
# Measured values on the axes.
# No training. These are definitions.

PROPERTY_VECTORS = {
    # Core Ontology (Axiomatic Properties)
    "existencia": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "entidad":    [1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],  # Synonym-ish
    "nada":       [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    
    "verdad":     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "falsedad":   [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    
    "orden":      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "caos":       [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    
    "acción":     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "estasis":    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    
    "conexión":   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "aislamiento":[0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    
    "mente":      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "objeto":     [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    
    "tiempo":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "pasado":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    
    # Discovery Property
    "intenso":    [0.0, 0.0, 0.8, 0.8, 0.0, 0.0, 0.0], # High Order + High Action

    # Life & Biology
    "vivo":     [1.0, 0.0, 0.0, 0.7, 0.3, 0.4, 0.6],  # Entity + Action + Time(growth)
    "animal":   [1.0, 0.0, 0.3, 0.5, 0.2, 0.1, 0.2],  # Entity + Action
    "humano":   [1.0, 0.0, 0.5, 0.5, 0.6, 0.9, 0.4],  # Entity + High Mind
    
    # Ethics & Truth
    "verdad":   [0.0, 1.0, 0.6, 0.0, 0.2, 0.2, 0.1],  # Truth + Order
    "mentira":  [0.0, -1.0, -0.6, 0.0, -0.2, -0.2, -0.1], # Anti-Truth + Chaos
    "bien":     [0.0, 0.5, 0.8, 0.2, 0.6, 0.0, 0.1],  # Order + Connection
    "mal":      [0.0, -0.5, -0.8, -0.2, -0.6, 0.0, -0.1], # Chaos + Isolation
    
    # Qualities
    "fiel":     [0.0, 0.4, 0.2, 0.1, 0.6, 0.3, 0.0],  # Truth + Connection
    "azul":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Neutral (needs sensory axis? mapped to Order usually)
    
    # Logic
    "si":       [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    "no":       [0.0, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# 3. LEXICAL DEFINITIONS (The 'Dictionary')
# This maps words to their component properties.
LEXICAL_DEFINITIONS = {
    # Animals
    "perro":    ["entidad", "vivo", "animal", "fiel"],
    "lobo":     ["entidad", "vivo", "animal"],  # Same but lacking 'fame/faith'? Or add 'wild'?
    "gato":     ["entidad", "vivo", "animal"],
    "persona":  ["entidad", "vivo", "humano"],
    
    # Abstract
    "verdad":   ["verdad"],
    "mentira":  ["mentira"],
    "amor":     ["bien", "fiel", "vivo"],
    
    # Affirmation
    "sí":       ["si", "verdad"],
    "no":       ["no", "mentira"],
    
    # Discovery Demo: Intensifiers
    # We define them explicitly using a "magnitud" or "intenso" axiom logic
    # Assume "grande" = Entity + Order
    # Assume "enorme" = Entity + Order + Order (Double Order/Magnitude?) OR specific 'intenso' prop which we add
    "grande":   ["entidad", "orden"],
    "enorme":   ["entidad", "orden", "intenso"], 
    
    # Better: Add 'intenso' property to PROPERTY_VECTORS temporary for this? 
    # Or reuse existing. Let's use 'acción' + 'caos' as high energy modifier.
    
    # Pairs for discovery:
    # 1. Bueno -> Excelente
    "bueno":    ["bien"],
    "excelente":["bien", "intenso"], # High "vibration"
    
    # 2. Malo -> Pésimo
    "malo":     ["mal"],
    "pésimo":   ["mal", "intenso"], # Same modifier pattern: Orden+Vivo?
    
    # 3. Grande -> Enorme
    # "grande":   ["entidad", "orden"] defined above
    # "enorme":   ["entidad", "orden", "orden", "vivo"], # Consistent modifier
    
    # 4. Target for Prediction: Rápido -> Veloz/Fugaz
    "rápido":   ["acción"],
    # System should predict: Acción + Intenso
    # Security Demo (Distinct Identities)
    "gerente":  ["entidad", "orden", "verdad", "mente"], # High authority structure
    "hacker":   ["entidad", "caos", "mentira", "aislamiento"], # Anti-structure
    
}



class SemanticAlgebra:
    """
    Algebraic Engine for Semantic Calculation.
    Zero dependencies. Pure list math.
    """
    
    @staticmethod
    def normalize(v: List[float]) -> List[float]:
        """Normalize vector to unit length"""
        norm_sq = sum(x*x for x in v)
        if norm_sq < 1e-9:
            return v
        norm = math.sqrt(norm_sq)
        return [x / norm for x in v]
    
    @staticmethod
    def compose(properties: List[str]) -> List[float]:
        """
        Algebraic composition: Sum of property vectors.
        v = norm(sum(p_i))
        """
        acc = [0.0] * DIM
        
        for p in properties:
            if p not in PROPERTY_VECTORS:
                # For now, ignore unknown or raise? 
                # Strict mode: raise.
                # Industrial mode: Warning + skip?
                # Let's Skip for robustness in demo
                print(f"Warning: Property '{p}' unknown")
                continue
                
            vec = PROPERTY_VECTORS[p]
            for i in range(DIM):
                acc[i] += vec[i]
                
        return SemanticAlgebra.normalize(acc)
    
    @staticmethod
    def value_of(word: str) -> List[float]:
        """Get calculated value of a word"""
        w = word.lower()
        if w in LEXICAL_DEFINITIONS:
            return SemanticAlgebra.compose(LEXICAL_DEFINITIONS[w])
        
        # Fallback: if it's a raw property
        if w in PROPERTY_VECTORS:
            return SemanticAlgebra.normalize(PROPERTY_VECTORS[w])
            
        # D8 Industrial Panic: Unknown word
        # Option 1: Return Zero (Blind)
        # Option 2: Fallback to Hash (Encoder v1)
        # We return Zero to force Definition-First approach
        return [0.0] * DIM

    @staticmethod
    def distance(a: List[float], b: List[float]) -> float:
        """Euclidean distance in 7D semantic space"""
        dist_sq = sum((a[i] - b[i])**2 for i in range(DIM))
        return math.sqrt(dist_sq)


if __name__ == "__main__":
    print("CMFO D8-CORE: Semantic Algebra")
    print("==============================")
    
    # Calculate
    v_perro = SemanticAlgebra.value_of("perro")
    v_lobo = SemanticAlgebra.value_of("lobo")
    v_verdad = SemanticAlgebra.value_of("verdad")
    v_mentira = SemanticAlgebra.value_of("mentira")
    
    print(f"Vector(perro): {[round(x, 2) for x in v_perro]}")
    
    # Verify Distances
    d_dog_wolf = SemanticAlgebra.distance(v_perro, v_lobo)
    d_dog_truth = SemanticAlgebra.distance(v_perro, v_verdad)
    d_truth_lie = SemanticAlgebra.distance(v_verdad, v_mentira)
    
    print("\nDistances (D8 Validation):")
    print(f"Perro <-> Lobo:    {d_dog_wolf:.4f}  (Expected: Small)")
    print(f"Perro <-> Verdad:  {d_dog_truth:.4f} (Expected: Large)")
    print(f"Verdad <-> Mentira: {d_truth_lie:.4f} (Expected: Large/Opposite)")
    
    # Assertions
    if d_dog_wolf < d_dog_truth:
        print("[PASS] Semantic Closeness (Biology vs Abstract)")
    else:
        print("[FAIL] Semantic Closeness")
        
    if d_truth_lie > 1.5:
        print("[PASS] Semantic Opposition")
    else:
        print("[FAIL] Semantic Opposition")
