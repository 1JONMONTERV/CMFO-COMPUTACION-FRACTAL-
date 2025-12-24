
import sys
import os
import random
import re
import time

# Add root directory to path to allow 'from cortex import ...'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Also bindings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../bindings/python')))

from cortex.memory import FractalMemory
from cortex.encoder import FractalEncoder
from cortex.corpus_data import CORPUS_TEXT
from cmfo.core.structural import FractalVector7
from cmfo.compiler.jit import FractalJIT

class DeepDreamer:
    """
    Autonomous Cognitive Engine.
    Processes a corpus, extracts concepts, and enters a recursive dream loop.
    """
    def __init__(self):
        self.memory = FractalMemory()
        self.encoder = FractalEncoder()
        self.dummy_h = [0.0] * 7
        self.concepts = []
        
        # Load Corpus
        self._ingest_corpus()
        
        # Short-term memory (The "Residual Dream")
        self.residual_field = [0.0] * 7

    def _ingest_corpus(self):
        print("[*] Ingesting Corpus...")
        # Simple extraction: Capitalized words and Nouns (heuristic)
        words = re.findall(r'\b[A-Za-z][a-z]+\b', CORPUS_TEXT)
        # Filter mostly interesting words (length > 3)
        interesting = set([w for w in words if len(w) > 3])
        self.concepts = list(interesting)
        print(f"[*] Extracted {len(self.concepts)} concepts (e.g., {self.concepts[:5]}).")
        
        # Index them all into memory so we can "recognize" them later
        for c in self.concepts:
            self.memory.learn(c, "Concept from Corpus")

    def rapid_eye_movement(self, cycles=10):
        print(f"\n==================================================")
        print(f"      ENTERING R.E.M. SLEEP ({cycles} CYCLES)       ")
        print(f"==================================================")
        
        history = []
        
        for i in range(cycles):
            # 1. Stochastic Selection
            # Pick 2 random concepts
            inputs = random.sample(self.concepts, 2)
            input_str = "-".join(inputs)
            
            print(f"\n[CYCLE {i+1}] Dreaming of: {inputs}")
            
            # 2. Physics Transmutation
            combined_vec = self.encoder.encode(input_str)
            # Mix with residual field (Previous dream influences this one)
            mixed_state = []
            for j in range(7):
                mixed_state.append(combined_vec.v[j] + self.residual_field[j] * 0.1)
                
            phi = 0.5 + (sum(abs(x) for x in mixed_state[:3]) % 2.0)
            lam = 0.01 + (sum(abs(x) for x in mixed_state[3:]) % 0.49)
            
            # 3. JIT Simulation
            v = FractalVector7.symbolic('v')
            eq = v * phi - (v * v * lam)
            
            current_state = list(combined_vec.v) # Start from the concept seed
            
            # Evolve
            for _ in range(30):
                res = FractalJIT.compile_and_run(eq._node, current_state, self.dummy_h)
                current_state = [x for row in res for x in row]
                # Normalize
                norm = sum(x**2 for x in current_state)**0.5
                if norm > 5.0: current_state = [x/norm for x in current_state]
            
            # 4. Update Residual (The Dream changes the Dreamer)
            self.residual_field = current_state
            
            # 5. Interpretation
            final_vec = FractalVector7(current_state)
            
            # Finding resonance in Memory
            best_match = None
            best_dist = 999.0
            
            for k in self.memory.keys:
                stored = self.memory.vectors[k]
                # Don't match the inputs themselves
                if stored['name'] in inputs: continue
                
                dist = self.encoder.conceptual_distance(final_vec, stored['vec_obj'])
                if dist < best_dist:
                    best_dist = dist
                    best_match = stored['name']
            
            print(f"    --> Manifests as: **{best_match.upper()}** (Dist: {best_dist:.2f})")
            history.append(f"{inputs[0]} + {inputs[1]} = {best_match}")
            
            # Small delay to simulate thought process
            # time.sleep(0.1) 
            
        print("\n[AWAKENING]")
        print("Dream Log:")
        for entry in history:
            print(f"  - {entry}")

if __name__ == "__main__":
    dreamer = DeepDreamer()
    dreamer.rapid_eye_movement(cycles=20)
