"""
CMFO D17: Seed Mathematics Vocabulary
=====================================
Formally defines the 'Matematica' domain.
The foundation of proof, allowing abstract concepts like Infinity.
"""

import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from cmfo.ontology.vocab import VocabularyManager

def seed_math():
    mgr = VocabularyManager()
    
    # 1. Define the Domain Physics for MATH
    # Math is the most abstract domain.
    # Ex: True, Ord: True, Time: Irrelevant (Eternal truths)?
    
    config = {
        "axioms": [
            "lógico",           # Consistent
            "abstracto",        # Not physical
            "formal",           # Syntactically defined
            "infinito"          # Allowed (Unlike CS often)
        ],
        "allowed_ops": [
            "implicación",
            "negación",
            "inducción",
            "deducción",
            "abstracción"
        ],
        "forbidden": [
            "empírico",         # Math is not experimental science
            "aproximado",       # (In pure math, 1 != 0.999...)
            "subjetivo",
            "fe"
        ],
        "vector_modifiers": {
            "5": 0.1,  # Mente: Cartesian Dualism? Keep low.
            "6": 0.0,  # Tiempo: Math is usually timeless (Eternalism)
            "3": 0.0   # Acción: Math is static truth, CS is dynamic Process.
        },
        "terms": [
            {"term": "teorema", "def": "Proposición demostrada lógicamente"},
            {"term": "lema", "def": "Proposición auxiliar"},
            {"term": "prueba", "def": "Secuencia de pasos deductivos"},
            {"term": "axioma", "def": "Verdad auto-evidente o asumida"},
            {"term": "infinito", "def": "Cardinalidad sin límite"}
        ]
    }
    
    print("Seeding Domain: MATEMATICA...")
    mgr.define_domain("matematica", config)
    
    # Verification
    vocab = mgr.get_context("matematica")
    if vocab:
        print(f"Domain Loaded: {vocab.domain}")
        # Test Projection: Time shoud be 0 (Timeless truth)
        vec = [1.0] * 7
        proj = vocab.get_projection(vec)
        print(f"Projection Test (Time Axis 6): {proj[6]}")

if __name__ == "__main__":
    seed_math()
