import sys
import os
import numpy as np
import unittest

# Ensure we can load local package
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))
from cmfo.core.matrix import T7Matrix

class TestNativeCorrectness(unittest.TestCase):
    def test_evolve_accuracy(self):
        print("\nTesting Native Engine Accuracy...")
        
        # Setup similar to benchmark
        vec = np.random.rand(7)
        mat_np = np.eye(7) # Identity for native default
        
        # 1. NumPy Reference
        v_ref = vec.copy()
        # Single step evolution: v = sin(M @ v)
        # Native T7Matrix is Identity by default
        # So reference is v = sin(I @ v) = sin(v)
        v_ref = np.sin(mat_np @ v_ref)
        
        # 2. Native Evolution
        try:
            # Use Identity to match mat_np = np.eye(7)
            mat_obj = T7Matrix.identity()
            # If native lib is missing, this should now work via Python fallback
                
            v_native = mat_obj.evolve_state(vec, steps=1)
            
            # 3. Compare
            # Use real part for comparison (imaginary should be 0 if input is real and matrix is identity)
            # Wait, sin(real) is real. 
            # Our native engine does complex math.
            # v = sin(v). If v is real, result is real.
            
            diff = np.linalg.norm(v_native - v_ref)
            print(f"Difference (State): {diff:.6e}")
            
            np.testing.assert_allclose(v_native.real, v_ref.real, rtol=1e-5, atol=1e-8)
            print("p.tests.assert_allclose PASSED")
            
        except Exception as e:
            self.fail(f"Native execution failed: {e}")

if __name__ == '__main__':
    unittest.main()
