"""
CMFO D6: Data Collection for Calibration
=========================================
Generate diverse decision data for attractor calibration.

This script simulates realistic decision scenarios to collect
enough data for geometric calibration.
"""

import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from cmfo.decision.enhanced_engine import EnhancedDecisionEngine, Context
from cmfo.decision.memory import FractalMemory
import random


def generate_training_data(num_samples: int = 200):
    """Generate diverse decision data for calibration"""
    print("=" * 70)
    print(f"  GENERATING {num_samples} TRAINING DECISIONS")
    print("=" * 70)
    
    memory = FractalMemory(dream_file="calibration_dreams.jsonl")
    engine = EnhancedDecisionEngine(memory=memory)
    
    # Decision templates by intent
    templates = {
        'confirm': [
            ([0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0], {}),
            ([0.75, 0.25, 0.05, 0.0, 0.0, 0.0, 0.0], {}),
            ([0.85, 0.15, 0.12, 0.0, 0.0, 0.0, 0.0], {}),
            ([0.78, 0.22, 0.08, 0.0, 0.0, 0.0, 0.0], {}),
        ],
        'correct': [
            ([-0.7, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0], {"correction": "error detectado"}),
            ([-0.8, 0.4, 0.3, 0.1, 0.0, 0.0, 0.0], {"correction": "valor incorrecto"}),
            ([-0.65, 0.55, 0.15, 0.0, 0.0, 0.0, 0.0], {"correction": "dato erróneo"}),
            ([-0.75, 0.45, 0.25, 0.05, 0.0, 0.0, 0.0], {"correction": "inconsistencia"}),
        ],
        'question': [
            ([0.2, 0.1, 0.8, 0.5, 0.2, 0.0, 0.0], {"entity": "término", "options": "A o B"}),
            ([0.15, 0.15, 0.75, 0.45, 0.25, 0.0, 0.0], {"entity": "concepto", "options": "X o Y"}),
            ([0.1, 0.2, 0.85, 0.55, 0.15, 0.0, 0.0], {"entity": "valor", "options": "1 o 2"}),
            ([0.25, 0.05, 0.7, 0.4, 0.3, 0.0, 0.0], {"entity": "opción", "options": "si o no"}),
        ],
        'conflict': [
            ([-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0], {"previous": "anterior", "explanation": "contradicción"}),
            ([-0.6, -0.4, 0.6, 0.4, 0.0, 0.0, 0.0], {"previous": "previo", "explanation": "inconsistente"}),
            ([-0.45, -0.55, 0.55, 0.45, 0.0, 0.0, 0.0], {"previous": "pasado", "explanation": "opuesto"}),
            ([-0.55, -0.45, 0.45, 0.55, 0.0, 0.0, 0.0], {"previous": "antes", "explanation": "conflicto"}),
        ]
    }
    
    # Generate samples
    samples_per_intent = num_samples // len(templates)
    
    for intent, intent_templates in templates.items():
        print(f"\nGenerating {samples_per_intent} samples for {intent}...")
        
        for i in range(samples_per_intent):
            # Pick random template
            base_vector, slots = random.choice(intent_templates)
            
            # Add noise (±0.1 on each dimension)
            noisy_vector = [
                v + random.uniform(-0.1, 0.1)
                for v in base_vector
            ]
            
            # Optional context
            context = None
            if random.random() > 0.5:
                context = Context(
                    vectors=[[random.uniform(-0.5, 0.5) for _ in range(7)]],
                    sources=["context.txt"],
                    weights=[1.0]
                )
            
            # Make decision (stores in memory automatically)
            response, proof = engine.decide(noisy_vector, context, slots)
            
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{samples_per_intent} completed")
    
    # Stats
    stats = memory.stats()
    print("\n" + "=" * 70)
    print("  DATA COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\nTotal decisions: {stats['total']}")
    print(f"By intent: {stats['by_intent']}")
    print(f"Avg confidence: {stats['avg_confidence']}")
    print(f"\nData saved to: calibration_dreams.jsonl")
    
    return memory


if __name__ == "__main__":
    # Generate 200 samples
    memory = generate_training_data(200)
    
    print("\n" + "=" * 70)
    print("  READY FOR CALIBRATION")
    print("=" * 70)
    print("\nNext step:")
    print("  python cmfo/decision/calibration.py")
