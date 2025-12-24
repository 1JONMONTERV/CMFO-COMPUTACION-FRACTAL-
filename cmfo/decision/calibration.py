"""
CMFO D6: Human-Grounded Attractor Calibration
==============================================
Attractors are MENTAL POSTURES, not mathematical clusters.

Each attractor represents a stable human cognitive stance:
- Coherent affirmation
- Factual error
- Direct contradiction
- Honest ambiguity
- Reference request

Calibrated with 5-10 real human language examples per posture.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import json


# Human Mental Postures (NOT technical labels)
HUMAN_POSTURES = {
    "coherent_affirmation": {
        "label": "Afirmación coherente",
        "description": "Confirmar algo que es correcto y consistente",
        "examples": [
            "Sí, París es la capital de Francia",
            "Correcto, 2+2 es 4",
            "Exacto, el agua hierve a 100°C",
            "Así es, la Tierra gira alrededor del Sol",
            "Cierto, Python es un lenguaje de programación"
        ],
        "intent": "confirm"
    },
    
    "factual_error": {
        "label": "Error factual común",
        "description": "Corregir un error objetivo y claro",
        "examples": [
            "No, Madrid es la capital de España, no Barcelona",
            "Incorrecto, 2+2 es 4, no 5",
            "Error: el agua hierve a 100°C, no a 50°C",
            "Falso, la Tierra gira alrededor del Sol, no al revés",
            "No es así, Python no es un sistema operativo"
        ],
        "intent": "correct"
    },
    
    "direct_contradiction": {
        "label": "Contradicción directa",
        "description": "Señalar inconsistencia con lo dicho antes",
        "examples": [
            "Antes dijiste que era azul, ahora dices rojo",
            "Esto contradice lo que mencionaste hace un momento",
            "Eso es opuesto a tu afirmación anterior",
            "Cambiaste de postura sin explicar por qué",
            "Esto no coincide con lo que establecimos antes"
        ],
        "intent": "conflict"
    },
    
    "honest_ambiguity": {
        "label": "Ambigüedad honesta",
        "description": "Pedir aclaración cuando algo no está claro",
        "examples": [
            "¿Te refieres a la ciudad o al país?",
            "No está claro si hablas de temperatura o presión",
            "¿Cuál de los dos significados?",
            "Eso puede interpretarse de dos formas distintas",
            "Necesito que especifiques a qué te refieres"
        ],
        "intent": "question"
    },
    
    "reference_request": {
        "label": "Petición de referencia",
        "description": "Remitir a algo ya mencionado",
        "examples": [
            "Como dijimos antes sobre este tema",
            "Volviendo a lo que mencionaste",
            "Esto se relaciona con lo anterior",
            "Ya hablamos de esto previamente",
            "Retomando el punto que tocaste"
        ],
        "intent": "reference"
    }
}


@dataclass
class HumanExample:
    """Single human language example"""
    text: str
    posture: str  # Human-readable posture name
    intent: str   # Technical intent (for mapping)


class HumanGroundedCalibrator:
    """
    Calibrates attractors from human language examples.
    
    NOT from synthetic data - from real human expressions.
    """
    
    def __init__(self):
        self.examples = []
        self.load_human_examples()
    
    def load_human_examples(self):
        """Load human language examples"""
        for posture_key, posture_data in HUMAN_POSTURES.items():
            for example_text in posture_data["examples"]:
                self.examples.append(HumanExample(
                    text=example_text,
                    posture=posture_data["label"],
                    intent=posture_data["intent"]
                ))
        
        print(f"Loaded {len(self.examples)} human examples")
        print(f"Postures: {list(HUMAN_POSTURES.keys())}")
    
    def encode_example(self, text: str) -> List[float]:
        """
        Encode human text to semantic vector.
        
        Uses deterministic FractalEncoder.
        """
        try:
            from .encoder import FractalEncoder
            encoder = FractalEncoder()
            return encoder.encode(text)
        except ImportError:
            # Local import fallback
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from encoder import FractalEncoder
            encoder = FractalEncoder()
            return encoder.encode(text)
    
    def calibrate_from_human_examples(self) -> Dict:
        """
        Calibrate attractors from human language examples.
        
        Returns dict of posture -> attractor spec
        """
        from collections import defaultdict
        import math
        
        PHI = 1.6180339887
        
        # Group examples by intent
        by_intent = defaultdict(list)
        for example in self.examples:
            vector = self.encode_example(example.text)
            by_intent[example.intent].append({
                'vector': vector,
                'text': example.text,
                'posture': example.posture
            })
        
        # Calculate centroids
        attractors = {}
        for intent, examples in by_intent.items():
            vectors = [e['vector'] for e in examples]
            
            # Geometric centroid
            centroid = [0.0] * 7
            for vec in vectors:
                for i in range(7):
                    centroid[i] += vec[i]
            for i in range(7):
                centroid[i] /= len(vectors)
            
            # Calculate radius (max distance)
            def d_phi(x, y):
                dist_sq = sum(PHI**i * (x[i]-y[i])**2 for i in range(7))
                return math.sqrt(dist_sq)
            
            distances = [d_phi(v, centroid) for v in vectors]
            radius = max(distances) if distances else 0.5
            
            # Find human label
            posture_label = examples[0]['posture']
            
            attractors[intent] = {
                'intent': intent,
                'posture': posture_label,
                'centroid': centroid,
                'radius': radius,
                'samples': len(examples),
                'example_texts': [e['text'] for e in examples[:3]]  # Keep 3 examples
            }
            
            print(f"\nCalibrated '{posture_label}':")
            print(f"  Intent: {intent}")
            print(f"  Samples: {len(examples)}")
            print(f"  Centroid: {[round(x, 3) for x in centroid[:3]]}...")
            print(f"  Radius: {radius:.4f}")
            print(f"  Examples:")
            for ex in examples[:2]:
                print(f"    - \"{ex['text']}\"")
        
        return attractors
    
    def save_human_calibration(self, attractors: Dict, output_file: str = "attractors_human_v1.json"):
        """Save human-calibrated attractors"""
        output = {
            "version": "1.0-human",
            "calibration_method": "human_language_examples",
            "description": "Attractors calibrated from real human language, not synthetic data",
            "postures": HUMAN_POSTURES,
            "attractors": attractors
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Saved human-calibrated attractors to {output_file}")


if __name__ == "__main__":
    print("=" * 70)
    print("  CMFO D6: Human-Grounded Calibration")
    print("  Attractors as Mental Postures")
    print("=" * 70)
    
    calibrator = HumanGroundedCalibrator()
    
    print("\n" + "=" * 70)
    print("  CALIBRATING FROM HUMAN EXAMPLES")
    print("=" * 70)
    
    attractors = calibrator.calibrate_from_human_examples()
    
    calibrator.save_human_calibration(attractors)
    
    print("\n" + "=" * 70)
    print("  HUMAN-GROUNDED CALIBRATION COMPLETE")
    print("=" * 70)
    print("\nAttractors are now MENTAL POSTURES, not clusters.")
    print("Each represents a stable human cognitive stance.")
    print("\nNext: Update enhanced_engine.py to load attractors_human_v1.json")
