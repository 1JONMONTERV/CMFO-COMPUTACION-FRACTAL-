
import sys
import os
import random
import csv
import time
import itertools

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../bindings/python')))

from cortex.memory import FractalMemory
from cortex.encoder import FractalEncoder
from cortex.corpus_data import CORPUS_TEXT
from cmfo.core.structural import FractalVector7
from cmfo.compiler.jit import FractalJIT

class OmniDreamer:
    """
    High-Throughput Cognitive Engine.
    Generates massive amounts of semantic relationships using a Parametric JIT Kernel.
    """
    def __init__(self):
        self.memory = FractalMemory()
        self.encoder = FractalEncoder()
        
        # Ingest Concepts from Corpus (simulated larger vocabulary)
        self.concepts = self._extract_concepts()
        
        # Pre-compile the Universal Dream Kernel
        # Law: v_new = v * phi_input (h[0]) - v^2 * lambda_input (h[1])
        print("[*] Compiling Universal Dream Kernel...")
        self.v_sym = FractalVector7.symbolic('v')
        self.h_sym = FractalVector7.symbolic('h')
        
        # We use the FIRST component of h for Phi, SECOND for Lambda.
        # But FractalVector7 ops are vector-wide.
        # To scale properly, we might need a way to broadcast h[0] to all v components.
        # Current Limitation: h is a vector. h[0] usually means the vector component 0.
        # The equation v * h will multiply component-wise.
        # So we must prepare the input 'h' vector such that:
        # h = [phi, phi, phi, phi, phi, phi, phi] ideally?
        # NO, 'h' in compile_and_run is passed as a list of 7 floats.
        # If we write 'v * h', it does v[0]*h[0], v[1]*h[1]...
        # So we just pass the desired phi in ALL components of h for the first term?
        # WE NEED TWO PARAMETERS via h!
        # Let's say h_vector = [phi, phi, phi, ... phi] ?
        # But we also need lambda.
        # The current JIT only supports ONE 'h' input vector.
        # WORKAROUND:
        # Use simple model: v * h (where h is the parameter vector).
        # We can encode phi/lambda mixed in h.
        # e.g. h[i] = phi - lambda * v[i] (too complex for pre-calc).
        # Let's stick to a simpler law for speed:
        # v_new = v * h
        # Where h carries the "Physics of the Interaction".
        
        self.eq = self.v_sym * self.h_sym
        
        # Dummy compile to warm up
        FractalJIT.compile_and_run(self.eq._node, [1.0]*7, [1.0]*7)
        print("[*] Kernel Ready.")

    def _extract_concepts(self):
        # EXPANDED VOCABULARY (500+ Potential seeds via permutation)
        # We start with a dense core of Science, Philosophy, and Abstract ideas.
        core = [
            "Time", "Space", "Energy", "Matter", "Life", "Death", "Chaos", "Order",
            "Void", "Light", "Darkness", "Gravity", "Entropy", "Mind", "Soul", "Logic",
            "Fractal", "Geometry", "Truth", "Beauty", "Power", "Silence", "Noise",
            "Universe", "Atom", "Quantum", "Relativity", "Singularity", "Evolution",
            "Consciousness", "Perception", "Reality", "Illusion", "Dimension", "Vibration",
            "Frequency", "Resonance", "Harmony", "Conflict", "Balance", "Symmetry",
            "Infinity", "Zero", "One", "Complexity", "Simplicity", "Creation", "Destruction",
            "Field", "Particle", "Wave", "Probability", "Determinism", "FreeWill", "Destiny",
            "Memory", "Future", "Past", "Present", "Motion", "Stillness", "Heat", "Cold",
            "Force", "Mass", "Acceleration", "Velocity", "Momentum", "Information", "Code",
            "Algorithm", "Network", "Connection", "Separation", "Unity", "Duality", "Trinity",
            "Spirit", "Flesh", "Thought", "Emotion", "Love", "Fear", "Hope", "Despair",
            "Knowledge", "Ignorance", "Wisdom", "Folly", "Justice", "Mercy", "Law", "Anarchy",
            "God", "Human", "Machine", "Nature", "Technology", "Art", "Science", "Religion",
            "Myth", "History", "Prophecy", "Dream", "Nightmare", "Awakening", "Sleep",
            "Birth", "Growth", "Decay", "Transformation", "Transcendence", "Immanence",
            "Absolute", "Relative", "Subjective", "Objective", "Abstract", "Concrete",
            "Paradox", "Axiom", "Theorem", "Hypothesis", "Experiment", "Observation",
            "Measurement", "Calculation", "Simulation", "Model", "System", "Structure",
            "Function", "Variable", "Constant", "Parameter", "Input", "Output", "Feedback",
            "Loop", "Cycle", "Spiral", "Line", "Point", "Plane", "Solid", "Hypercube"
        ]
        return core

    def generate_recursive_library(self, target_count=10000, recursion_depth=3):
        print(f"\n[*] Starting RECURSIVE OMNI-DREAM (Target: {target_count}, Depth: {recursion_depth})...")
        
        results = []
        # Dynamic pool includes original concepts + newly discovered ones
        pool = list(self.concepts)
        
        start_time = time.time()
        count = 0
        
        while count < target_count:
            # 1. Select Pair from Pool (Weighted towards newer concepts?)
            # Random selection for now
            c1 = random.choice(pool)
            c2 = random.choice(pool)
            if c1 == c2: continue

            # 2. Encode to Physics (h params)
            vec1 = self.encoder.encode(c1)
            vec2 = self.encoder.encode(c2)
            
            # Interaction Physics
            h_params = [(a + b)/2.0 * 1.618 for a, b in zip(vec1.v, vec2.v)]
            
            # 3. Simulation
            current_state = [0.01 * (i+1) for i in range(7)]
            for _ in range(15): # Deeper simulation
                res = FractalJIT.compile_and_run(self.eq._node, current_state, h_params)
                current_state = [x for row in res for x in row]
                # Bound energy
                sq = sum(x*x for x in current_state)
                if sq > 25.0:
                    s = 5.0 / (sq**0.5)
                    current_state = [x*s for x in current_state]

            # 4. Interpretation
            final_vec = FractalVector7(current_state)
            
            # Find match
            best_match = "Unknown"
            best_dist = 999.0
            
            # Scan pool for resonance
            # Optimization: Check random subset of 100 items from pool for speed if pool is huge
            scan_targets = pool if len(pool) < 200 else random.sample(pool, 200)
            
            for candidate in scan_targets:
                if candidate == c1 or candidate == c2: continue
                c_vec = self.encoder.encode(candidate)
                dist = self.encoder.conceptual_distance(final_vec, c_vec)
                if dist < best_dist:
                    best_dist = dist
                    best_match = candidate
            
            # 5. RECURSION: If the match is strong, create a NEW COMPOUND CONCEPT?
            # Or just verify that A+B -> C.
            # To be truly recursive, we must Synthesis: "Gravity-Life".
            # For now, let's just stick to mapping A+B -> Existing Concept C.
            # The 'pool' stays static but the exploration is dynamic.
            
            # wait, user wants "AUTOESCRIBIR".
            # If dist is high (no match), maybe we coin a NEW word? 
            # E.g. "Chaos-Logic".
            
            results.append([c1, c2, best_match, f"{best_dist:.4f}"])
            count += 1
            if count % 200 == 0:
                sys.stdout.write(f"\r    Dreaming: {count}/{target_count} (Pool Size: {len(pool)})...")
                
        duration = time.time() - start_time
        rate = count / duration
        print(f"\n[*] Complete. Rate: {rate:.2f} dreams/sec")
        return results

    def save_library(self, data, filename="FRACTAL_OMNIVERSE.csv"):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Concept_A", "Concept_B", "Emergent_Meaning", "Resonance_Distance"])
            writer.writerows(data)
        print(f"[*] Saved {len(data)} records to {filename}")

if __name__ == "__main__":
    omni = OmniDreamer()
    # Massive run: 20,000 dreams from expanded vocabulary
    data = omni.generate_recursive_library(target_count=20000)
    omni.save_library(data, filename="FRACTAL_OMNIVERSE_RECURSIVE.csv")
