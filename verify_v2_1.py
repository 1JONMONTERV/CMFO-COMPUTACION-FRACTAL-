
import sys
import math
sys.path.insert(0, 'bindings/python')

from cmfo.core.structural import FractalVector7
from cmfo.core.gpu import Accelerator
from cmfo.core.matrix import T7Matrix

def test_vector_ops():
    print("Testing FractalVector7 Ops...")
    v = FractalVector7([1.0]*7)
    v2 = v * 2.0
    v3 = 3.0 * v
    
    assert v2.v[0] == 2.0, "Scalar mul failed"
    assert v3.v[0] == 3.0, "Reverse scalar mul failed"
    print("✅ Vector Ops Passed")

def test_gpu_virtual():
    print("Testing GPU Virtual Backend...")
    Accelerator._load_library() # Force reload
    if Accelerator._is_virtual:
        print("Confirmed: Using Virtual GPU")
        kernel = Accelerator.get_kernel("linear_7d")
        assert kernel is not None, "Kernel linear_7d not found"
        
        # Test virtual execution
        data = [[1.0]*7]
        res = kernel(data)
        assert len(res) == 1, "Virtual kernel failed result size"
        print("✅ Virtual GPU Passed")
    else:
        print("⚠️ Warning: Loaded Native Lib unexpectedly, skipping virtual test")

def test_matrix_epsilon():
    print("Testing Matrix Epsilon...")
    # T7Matrix logic is harder to test directly without precise numerical setup,
    # but we can check if it runs.
    m = T7Matrix.identity()
    v = [0.1]*7
    res = m.evolve_state(v, steps=1)
    print("✅ Matrix Evolve ran")

if __name__ == "__main__":
    test_vector_ops()
    test_gpu_virtual()
    test_matrix_epsilon()
