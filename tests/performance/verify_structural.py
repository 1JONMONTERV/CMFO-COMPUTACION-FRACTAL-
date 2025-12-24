
import sys
import os
import unittest

# Ensure we pick up local bindings
sys.path.insert(0, os.path.abspath("bindings/python"))

# Mock absence of numpy if needed, or just force backend
# We will inspect the T7Matrix class
from cmfo.core.matrix import T7Matrix
from cmfo.core.structural import FractalVector7, FractalMatrix7

class TestStructuralBackend(unittest.TestCase):
    def test_force_structural(self):
        print("Testing Pure Python Structural Backend...")
        
        # 1. Initialize Matrix
        m = T7Matrix()
        
        # Force backend switch (simulating ImportError of numpy)
        m.backend = "structural"
        m.python_matrix = FractalMatrix7.identity()
        
        print(f"Backend set to: {m.backend}")
        
        # 2. Create Vector (List, not numpy array)
        vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        # 3. Evolve
        # evolve_state should detect backend='structural' and return list/FractalVector
        v_next = m.evolve_state(vec, steps=1)
        
        print(f"Result type: {type(v_next)}")
        print(f"Result: {v_next}")
        
        # Check values manually
        # Identity matrix -> v_next = normalize(sin(v))
        import cmath
        import math
        
        # Calculate expected
        sin_v = [cmath.sin(x) for x in vec]
        norm = math.sqrt(sum(abs(x)**2 for x in sin_v))
        expected = [x/norm for x in sin_v]
        
        # Compare
        for i in range(7):
            diff = abs(v_next[i] - expected[i])
            self.assertLess(diff, 1e-9, f"Mismatch at index {i}")
            
        print("[OK] Structural Evolution Verified Correct.")

if __name__ == "__main__":
    unittest.main()
