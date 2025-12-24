
import sys
import os
# Add root directory to path to allow 'from cortex import ...'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Also bindings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../bindings/python')))

from cortex.memory import FractalMemory
from cortex.encoder import FractalEncoder
from cmfo.core.structural import FractalVector7
from cmfo.compiler.jit import FractalJIT

class FractalDreamer:
    """
    Cognitive Autopoiesis Engine.
    1. Consumes Ideas.
    2. Compiles a Universe based on those ideas (Re-compilation).
    3. Simulates the interaction.
    4. Definitions emerge from the chaos.
    """
    def __init__(self):
        self.memory = FractalMemory()
        self.encoder = FractalEncoder()
        self.dummy_h = [0.0] * 7
        
        # Load a Dictionary of Base Concepts to map results back to
        self.vocabulary = [
            "Chaos", "Order", "Energy", "Void", "Structure", 
            "Flow", "Stasis", "Life", "Entropy", "Light", 
            "Darkness", "Time", "Gravity", "Expansion"
        ]
        # Learn vocabulary
        print("[*] Dreamer Learning Vocabulary...")
        for word in self.vocabulary:
            self.memory.learn(word, "Base Concept")
            
    def transmute_words_to_physics(self, words):
        """
        Derives Phi and Lambda from the geometric hash of the input words.
        """
        combined = "".join(words)
        vec = self.encoder.encode(combined)
        
        # Extract scalar properties from the 7D vector
        # Use simple mapping for demo
        # Phi ~ Magnitude of first 3 dims
        # Lambda ~ Magnitude of last 3 dims
        
        mag_phi = sum(abs(x) for x in vec.v[:3])
        mag_lam = sum(abs(x) for x in vec.v[3:])
        
        # Map to valid physics range
        # Phi [0.5, 2.5]
        # Lambda [0.01, 0.5]
        
        phi = 0.5 + (mag_phi % 2.0)
        lam = 0.01 + (mag_lam % 0.49)
        
        return phi, lam

    def dream(self, input_ideas):
        print(f"\n--- DREAMING OF: {input_ideas} ---")
        
        # 1. Transmutation
        phi, lam = self.transmute_words_to_physics(input_ideas)
        print(f"    [1] Transmutation: The words became Physics.")
        print(f"        Phi (Expansion)    = {phi:.4f}")
        print(f"        Lambda (Decay)     = {lam:.4f}")
        
        # 2. Re-Compilation (The core request)
        print(f"    [2] JIT Re-Compilation: Generating Kernel for '{'-'.join(input_ideas)}'...")
        
        v = FractalVector7.symbolic('v')
        h = FractalVector7.symbolic('h')
        
        # The Kernel of the Dream
        # Law: v_new = v * phi - v^2 * lambda
        eq = v * phi - (v * v * lam) + (h * 0.0)
        
        # 3. Simulation (The Dream State)
        print(f"    [3] Simulation: Evolving the Concept Universe...")
        # Seed with the vector of the first word
        initial_vec = self.encoder.encode(input_ideas[0])
        current_state = list(initial_vec.v)
        
        # Run for 50 fractal generations
        for _ in range(50):
            res = FractalJIT.compile_and_run(eq._node, current_state, self.dummy_h)
            current_state = [x for row in res for x in row]
            # Normalize
            norm = sum(x**2 for x in current_state)**0.5
            if norm > 5.0: current_state = [x/norm for x in current_state]
            
        # 4. Interpretation (Wake Up)
        print(f"    [4] Interpretation: What has emerged?")
        final_vec = FractalVector7(current_state)
        
        # Query memory to find what concept matches this final state
        # We query against the base vocabulary
        results = self.memory.query("QUERY_PLAHOLDER", top_k=3) 
        # Hack: memory.query takes text string to encode. 
        # We need to query with a raw vector.
        # Let's add a raw_query method to memory or just iterate here.
        
        best_match = None
        best_dist = 999.0
        
        for k in self.memory.keys:
            stored = self.memory.vectors[k]
            dist = self.encoder.conceptual_distance(final_vec, stored['vec_obj'])
            if dist < best_dist:
                best_dist = dist
                best_match = stored['name']
                
        print(f"    [RESULT] The System Defines '{' + '.join(input_ideas)}' as:")
        print(f"             ---> **{best_match.upper()}**")
        print(f"             (Resonance Distance: {best_dist:.4f})")
        
        return best_match

if __name__ == "__main__":
    dreamer = FractalDreamer()
    
    # Experiment 1
    dreamer.dream(["Chaos", "Time"])
    
    # Experiment 2
    dreamer.dream(["Light", "Energy"])
    
    # Experiment 3
    dreamer.dream(["Void", "Structure"])
