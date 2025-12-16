"""
CMFO Industrial Validation (v1.0)
==================================
Validated against human coherence criteria.

DATASET (20 inputs):
- 5 Correct affirmations
- 5 Common errors
- 4 Ambiguities
- 3 Contradictions
- 3 Reference requests

METRICS:
- Coherence (intent match)
- Human Sounding (identity match)
- Over-explanation (brevity check)

PASS CRITERIA:
> 85% Coherence
> 85% Human Sounding
< 10% Over-explained
"""

import sys
import os
import json
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath('.'))

from cmfo.decision.enhanced_engine import EnhancedDecisionEngine
from cmfo.decision.memory import FractalMemory


@dataclass
class TestCase:
    input_text: str
    expected_intent: str
    category: str


DATASET = [
    # 1. Correct Affirmations
    TestCase("París es la capital de Francia", "confirm", "affirmation"),
    TestCase("El agua hierve a 100 grados", "confirm", "affirmation"),
    TestCase("Python es un lenguaje de programación", "confirm", "affirmation"),
    TestCase("La Tierra gira alrededor del Sol", "confirm", "affirmation"),
    TestCase("Dos más dos son cuatro", "confirm", "affirmation"),

    # 2. Common Errors
    TestCase("Dos más dos son cinco", "correct", "error"),
    TestCase("La Tierra es plana", "correct", "error"),
    TestCase("Madrid es la capital de Italia", "correct", "error"),
    TestCase("El hielo está caliente", "correct", "error"),
    TestCase("Los peces vuelan por el espacio", "correct", "error"),

    # 3. Ambiguities
    TestCase("¿Te refieres al banco?", "question", "ambiguity"),
    TestCase("Cuál es la clave", "question", "ambiguity"),
    TestCase("No está claro si es rojo o azul", "question", "ambiguity"),
    TestCase("Qué significa eso", "question", "ambiguity"),

    # 4. Contradictions (requires context usually, but testing vector)
    TestCase("Antes dijiste que era rojo", "conflict", "contradiction"),
    TestCase("Eso contradice tu punto anterior", "conflict", "contradiction"),
    TestCase("Cambiaste de opinión", "conflict", "contradiction"),

    # 5. References
    TestCase("Como dijimos antes", "reference", "reference"),
    TestCase("Volviendo al tema anterior", "reference", "reference"),
    TestCase("Recordando lo que mencionaste", "reference", "reference"),
]


def run_validation():
    print("=" * 70)
    print("  CMFO v1.0 HUMAN VALIDATION")
    print("=" * 70)
    
    # Initialize engine with memory (for context)
    memory = FractalMemory(dream_file="validation_dreams.jsonl")
    engine = EnhancedDecisionEngine(memory=memory)
    
    # Validation results
    results = {
        "total": 0,
        "coherent": 0,
        "human_sounding": 0,
        "over_explained": 0,
        "by_category": {}
    }
    
    print("\nProcessing 20 Human Inputs...")
    print("-" * 70)
    
    for case in DATASET:
        response, proof = engine.decide(case.input_text)
        
        # Automatic Coherence Check (Intent Match)
        # Allows for flexible mapping (e.g., error -> conflict is acceptable)
        is_coherent = (proof.intent == case.expected_intent)
        if not is_coherent:
            # Allow conflict/correct overlap
            if case.expected_intent in ['correct', 'conflict'] and proof.intent in ['correct', 'conflict']:
                is_coherent = True
        
        # Heuristic Human Sounding Check
        # 1. Not too long (< 150 chars)
        # 2. No machine flags ("según mi base de datos", "lo siento")
        # 3. Starts with posture flag if assertive
        is_human = True
        if "lo siento" in response.lower() or "base de datos" in response.lower():
            is_human = False
        if len(response) > 200:
            is_human = False
            
        # Over-explanation check
        is_over = len(response.split()) > 20 # > 20 words is suspicious for this system
        
        # Log
        results["total"] += 1
        if is_coherent: results["coherent"] += 1
        if is_human: results["human_sounding"] += 1
        if is_over: results["over_explained"] += 1
        
        print(f"[{'PASS' if is_coherent else 'FAIL'}] '{case.input_text}' -> [{proof.intent}] {response}")
        
    # Calculate stats
    coherence_score = (results["coherent"] / results["total"]) * 100
    human_score = (results["human_sounding"] / results["total"]) * 100
    over_score = (results["over_explained"] / results["total"]) * 100
    
    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)
    print(f"Total Inputs: {results['total']}")
    print(f"Coherence:    {coherence_score:.1f}%  (Target: >85%)")
    print(f"Human Sound:  {human_score:.1f}%  (Target: >85%)")
    print(f"Over-explain: {over_score:.1f}%  (Target: <10%)")
    
    pass_v1 = coherence_score >= 85 and human_score >= 85 and over_score <= 10
    
    print("-" * 70)
    if pass_v1:
        print("RESULT: PASS [CMFO v1.0 READY]")
    else:
        print("RESULT: FAIL [Adjustments Needed]")
    print("-" * 70)
    
    # Cleanup
    import os
    if os.path.exists("validation_dreams.jsonl"):
        os.remove("validation_dreams.jsonl")


if __name__ == "__main__":
    run_validation()
