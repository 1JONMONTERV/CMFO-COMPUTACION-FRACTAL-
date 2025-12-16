"""
CMFO D14: Seed Computation Vocabulary
=====================================
Formally defines the 'Computacion' domain logic.
Sets strict axioms and vector projections.
"""

import sys
import os

# Fix path
sys.path.insert(0, os.path.abspath('.'))

from cmfo.ontology.vocab import VocabularyManager

def seed_computation():
    mgr = VocabularyManager()
    
    # 1. Define the Domain Physics
    # In CS (Computation), certain axes of the 7D torus behave differently.
    # We explicitly suppress 'Spirit/Mind' (often Axis 5 in this ontology) because
    # algorithms are formal procedures, not conscious entities (yet).
    
    config = {
        "axioms": [
            "finito",           # Must terminate or be countable
            "efectivo",         # Must be computable (Effectiveness)
            "determinista",     # Same input -> Same output (in pure func)
            "discreto"          # Digital nature
        ],
        "allowed_ops": [
            "composición",
            "iteración",
            "recursión",
            "asignación"
        ],
        "forbidden": [
            "milagro",          # Non-causal event
            "revelación",       # Knowledge without data source
            "fe",               # Belief without proof
            "magia",            # Effect without mechanism
            "ambigüedad"        # Syntax error in formal systems
        ],
        "vector_modifiers": {
            "5": 0.0,  # Mente/Consciencia -> Irrelevant in pure algorithms
            "6": 1.0   # Tiempo -> Highly relevant (Complexity)
        },
        "terms": [
            {"term": "algoritmo", "def": "Secuencia finita de instrucciones bien definidas"},
            {"term": "variable", "def": "Espacio de memoria con nombre y valor mutable"},
            {"term": "función", "def": "Mapeo determinista de entrada a salida"},
            {"term": "bug", "def": "Discrepancia entre especificación y ejecución"}
        ]
    }
    
    print("Seeding Domain: COMPUTACION...")
    mgr.define_domain("computacion", config)
    
    # Verification
    vocab = mgr.get_context("computacion")
    if vocab:
        print(f"Domain Loaded: {vocab.domain}")
        print(f"Forbidden: {vocab.forbidden}")
        
        # Test Projection
        # Imagine a vector [1, 1, 1, 1, 1, 1, 1] (All ones)
        # It should become [1, 1, 1, 1, 1, 0, 1] (Mind zeroed)
        test_vec = [1.0] * 7
        proj_vec = vocab.get_projection(test_vec)
        print(f"Projection Test (Mind Axis 5): {proj_vec[5]}")

if __name__ == "__main__":
    seed_computation()
