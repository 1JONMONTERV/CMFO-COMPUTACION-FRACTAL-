
import unittest
import sys
import time
import math
sys.path.insert(0, '../bindings/python')
from cmfo.core.structural import FractalVector7
from cmfo.compiler.jit import FractalJIT
from cmfo.compiler.ir import symbol, constant, fractal_add, fractal_mul

class TestDeepStress(unittest.TestCase):

    def test_deep_graph_compilation(self):
        print("\n[Stress] 1. Deep Graph Compilation (100 Ops Chain)...")
        
        # 1. Build a massive graph
        v = FractalVector7.symbolic('v') # Arg 1
        h = FractalVector7.symbolic('h') # Arg 2 (Required by legacy bridge)
        
        current = v
        
        # Operation: Decay and Shift
        # v_new = v * 0.99 + 0.01
        # Repeated 50 times -> Huge nested graph
        DEPTH = 50 
        for i in range(DEPTH):
            # We add h*0 to force 'h' into the signature without affecting value
            current = current * 0.99 + 0.01 + (h * 0.0)
            
        # 2. Compile
        print(f"    Graph Depth: {DEPTH} layers")
        
        # Prepare Input Data (Real execution)
        input_data = [1.0] * 7
        dummy_h = [0.0] * 7 # Dummy input
        
        # For execution via JIT wrapper, we need to extract the node
        # Because current FractalVector7.compute() is not fully wired for context injection,
        # we manually invoke JIT
        
        start_t = time.time()
        # Compile and Run ONCE
        res = FractalJIT.compile_and_run(current._node, input_data, dummy_h)
        compile_time = time.time() - start_t
        
        print(f"    Compilation + First Run: {compile_time*1000:.2f} ms")
        self.assertIsNotNone(res)
        
        # 3. Validation
        # Calculate CPU expected
        # x = 1.0
        # for _ in range(50): x = x*0.99 + 0.01
        expected = 1.0
        for _ in range(DEPTH):
            expected = expected * 0.99 + 0.01
            
        gpu_val = res[0][0] # First component of result vector
        diff = abs(gpu_val - expected)
        print(f"    CPU: {expected:.6f} vs GPU: {gpu_val:.6f}")
        self.assertLess(diff, 1e-4, "Deep Calculation Diverged!")

    def test_thermal_load(self):
        print("\n[Stress] 2. Thermal Load (10,000 Iterations)...")
        # Simple graph, massive repetition
        v = FractalVector7.symbolic('v')
        h = FractalVector7.symbolic('h')
        expr = v + h
        
        input_v = [1.0] * 7
        input_h = [2.0] * 7
        
        ITERATIONS = 10000
        start_t = time.time()
        
        # This will use the cache after the first run
        for i in range(ITERATIONS):
            FractalJIT.compile_and_run(expr._node, input_v, input_h)
            
        total_time = time.time() - start_t
        tps = ITERATIONS / total_time
        print(f"    Total Time: {total_time:.2f}s")
        print(f"    Throughput: {tps:.0f} ops/sec")
        
        self.assertGreater(tps, 500, "Throughput below certification standard (500 ops/s)")

if __name__ == '__main__':
    unittest.main()
