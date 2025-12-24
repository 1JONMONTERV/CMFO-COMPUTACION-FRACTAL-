
"""
CMFO v3.0 FINAL DEMONSTRATION
=============================
Pipeline: Python IR -> CUDAGenerator -> Native JIT -> GPU Execution
"""

import sys
sys.path.insert(0, 'bindings/python')
import time
import cmfo # Initialize env

from cmfo.compiler.jit import FractalJIT
from cmfo.compiler.ir import *

def run_demo():
    print("\n" + "="*50)
    print("CMFO v3.0 NATIVE JIT EXECUTION PROTOCOL")
    print("="*50 + "\n")
    
    # 1. Define Fractal Operation (The "Sniper" Shot)
    # Formula: z = (v * v + 0.618)
    print("[1] Constructing Fractal Graph...")
    v = symbol('v')
    h = symbol('h') # Not used in this specific small test but required by signature
    
    term1 = fractal_mul(v, v)
    expr = fractal_add(term1, constant(0.618034))
    
    print(f"    Graph: {expr}")

    # 2. Prepare Data
    print("[2] Allocating Host Memory...")
    N_VECTORS = 10
    # Input: 10 vectors of 7D ones [1, 1, ..., 1]
    input_v = [1.0] * (N_VECTORS * 7)
    input_h = [0.0] * (N_VECTORS * 7) # Dummy
    
    # 3. Execute JIT
    print("[3] Launching 'The Sniper' (Compile & Run)...")
    start = time.time()
    
    try:
        results = FractalJIT.compile_and_run(expr, input_v, input_h)
        end = time.time()
        print(f"    Success! Execution Time: {end - start:.4f}s")
        
        # 4. Verify Result
        print("[4] Verifying Output...")
        # Expected: 1*1 + 0.618 = 1.618
        first_vector = results[0]
        print(f"    Vector[0]: {first_vector}")
        
        error = abs(first_vector[0] - 1.618034)
        if error < 1e-5:
            print(f"    ✅ ACCURACY CHECK PASSED (Error: {error:.2e})")
            print("\nMISSION ACCOMPLISHED: SYSTEM IS LIVE.")
        else:
            print(f"    ❌ ACCURACY CHECK FAILED. Expected ~1.618, Got {first_vector[0]}")
            
    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
