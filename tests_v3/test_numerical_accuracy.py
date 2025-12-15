
import sys
import math
import random
import unittest
sys.path.insert(0, '../bindings/python')
from cmfo.core.structural import FractalVector7
from cmfo.compiler.jit import FractalJIT

class TestNumericalAccuracy(unittest.TestCase):
    
    def setUp(self):
        # Generate random inputs
        self.v1_vals = [random.uniform(-10, 10) + 0j for _ in range(7)]
        self.v2_vals = [random.uniform(-10, 10) + 0j for _ in range(7)]
        self.v3_vals = [random.uniform(-10, 10) + 0j for _ in range(7)]
        
        self.v1_ref = FractalVector7(self.v1_vals)
        self.v2_ref = FractalVector7(self.v2_vals)
        self.v3_ref = FractalVector7(self.v3_vals)

    def verify_vectors(self, ref, jit_result, tol=1e-4):
        # JIT result currently comes as [ [f,f,f,f,f,f,f], ... ] or flat lists? 
        # depends on jit.py output format for single run.
        # compile_and_run returns a list of outputs (one per input slot).
        
        # If we passed 1 vector, we get [[...]].
        vec_out = jit_result[0] # Assuming list of list
        
        for i in range(7):
            cpu_val = ref.v[i].real # Only real comparison for now (Phase 1)
            gpu_val = vec_out[i]
            diff = abs(cpu_val - gpu_val)
            self.assertLess(
                diff, tol, 
                f"\nComponent {i} mismatch.\nCPU: {cpu_val}\nGPU: {gpu_val}\nDiff: {diff}"
            )

    def test_basic_addition(self):
        print("\n[Test] Addition (v1 + v2)")
        # CPU
        ref_res = self.v1_ref + self.v2_ref
        
        # GPU (Manually triggering JIT via exposed API for now as Auto-JIT is "Lazy Only")
        # In this test we use the raw JIT API to verify the kernel correctness, 
        # bypassing the Lazy Wrapper complexity for pure math verification.
        from cmfo.compiler.ir import symbol, fractal_add
        
        # Build IR
        v_node = symbol('v')
        h_node = symbol('h')
        expr = fractal_add(v_node, h_node)
        
        # Run JIT
        # Input expected as FLAT lists or List of List? jit.py accepts List[float]
        input_v = [x.real for x in self.v1_ref.v]
        input_h = [x.real for x in self.v2_ref.v]
        
        jit_res = FractalJIT.compile_and_run(expr, input_v, input_h)
        self.verify_vectors(ref_res, jit_res)

    def test_complex_expression(self):
        print("\n[Test] Complex Expr: v*0.5 + h")
        # CPU
        ref_res = self.v1_ref * 0.5 + self.v2_ref
        
        # GPU
        from cmfo.compiler.ir import symbol, fractal_add, fractal_mul, constant
        v = symbol('v')
        h = symbol('h')
        term1 = fractal_mul(v, constant(0.5))
        expr = fractal_add(term1, h)
        
        input_v = [x.real for x in self.v1_ref.v]
        input_h = [x.real for x in self.v2_ref.v]
        
        jit_res = FractalJIT.compile_and_run(expr, input_v, input_h)
        self.verify_vectors(ref_res, jit_res)

if __name__ == '__main__':
    print("========================================")
    print("  CMFO v3.1 JIT - NUMERICAL ACCURACY TEST")
    print("========================================")
    unittest.main()
