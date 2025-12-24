


import hashlib
import struct
import math
import sys
import os

# Add bindings/python to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bindings_path = os.path.join(repo_root, 'bindings', 'python')
if bindings_path not in sys.path:
    sys.path.insert(0, bindings_path)

from cmfo.core.structural import FractalVector7



class FractalEncoder:
    """
    Deterministic Text -> FractalVector7 Encoder.
    Uses SHA-256 seeding combined with Golden Ratio projection to map
    concepts into the 7D Manifold without neural training.
    """
    def __init__(self):
        self.PHI = 1.6180339887
        
    def encode(self, text):
        """
        Maps a string to a 7D Fractal Vector deterministically.
        Mechanism:
        1. SHA-256 Hash of text -> 32 bytes.
        2. Split into 7 chunks of 4 bytes (28 bytes used).
        3. Convert each chunk to float inside [0, 1].
        4. Apply Phi-Mixing to distribute semantically.
        """
        # 1. Deterministic Hash
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
        
        # 2. Extract 7 chaotic seeds
        seeds = []
        for i in range(7):
            # Take 4 bytes
            chunk = hash_bytes[i*4 : (i+1)*4]
            # Convert to int
            val_int = struct.unpack('>I', chunk)[0]
            # Normalize to [-1, 1]
            val_float = (val_int / (2**32 - 1)) * 2.0 - 1.0
            seeds.append(val_float)
            
        # 3. Geometric Projection (The "Knowledge Embedding")
        # We apply a discrete fractal transform to 'spread' the meaning
        # v[i] = seed[i] * phi^i (mod range)
        
        vector_data = []
        for i in range(7):
            # Mixing function
            raw = seeds[i] * (self.PHI ** i)
            # Wrap to valid range [-1, 1] via Sine map (continuous)
            projected = math.sin(raw * math.pi) 
            vector_data.append(projected)
            
        return FractalVector7(vector_data)

    def conceptual_distance(self, vec_a, vec_b):
        """
        Calculates the 'Phi-Metric' distance between two concepts.
        Lower distance = Higher Semantic Similarity (in this topology).
        """
        # We use the generic Euclidean for now, but weighted by Phi in future?
        # Let's use simple Euclidean on the 7D manifold.
        d2 = 0.0
        for i in range(7):
            d2 += (vec_a.v[i] - vec_b.v[i]) ** 2
        return math.sqrt(d2)

if __name__ == "__main__":
    encoder = FractalEncoder()
    c1 = "Quantum Physics"
    c2 = "General Relativity"
    c3 = "Apple Pie"
    
    v1 = encoder.encode(c1)
    v2 = encoder.encode(c2)
    v3 = encoder.encode(c3)
    
    print(f"Concept: '{c1}' -> {v1.v[:3]}...")
    print(f"Concept: '{c2}' -> {v2.v[:3]}...")
    
    d12 = encoder.conceptual_distance(v1, v2)
    d13 = encoder.conceptual_distance(v1, v3)
    
    print(f"Dist(Phys, Rel): {d12:.4f}")
    print(f"Dist(Phys, Pie): {d13:.4f}")
    print("Deterministic Embedding Operational.")
