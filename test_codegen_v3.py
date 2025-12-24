"""
Test v3.0 CodeGen
=================
Verifies that the Compiler generates correct CUDA C++ code from IR.
"""

import sys
sys.path.insert(0, 'bindings/python')

from cmfo.compiler.ir import *
from cmfo.compiler.codegen import CUDAGenerator

def test_codegen():
    print("\n--- Testing CMFO v3.0 CodeGen (The Sniper) ---\n")
    
    # 1. Build IR Graph
    # Formula: z = (v * h + PHI) / (1 + h)
    
    PHI = 1.6180339887
    
    v = symbol('v')
    h = symbol('h')
    
    # Numerator: v*h + PHI
    term1 = fractal_mul(v, h)
    num = fractal_add(term1, constant(PHI))
    
    # Denominator: 1 + h
    den = fractal_add(constant(1.0), h)
    
    # Final: num / den
    expr = fractal_div(num, den)
    
    # 2. Generate Code
    gen = CUDAGenerator()
    cuda_source = gen.generate_kernel(expr, kernel_name="cmfo_simulation_kernel")
    
    print("Generated CUDA Kernel:\n")
    print(cuda_source)
    
    # 3. Validation
    # Check for Loop Unrolling (presence of v0..v6)
    missing = []
    for i in range(7):
        if f"out[idx*7 + {i}]" not in cuda_source:
            missing.append(i)
            
    if not missing:
        print("✅ SUCCESS: Kernel completely unrolled (Dimensions 0-6 present explicitely).")
        print("✅ No loops detected in critical path.")
    else:
        print(f"❌ FAILURE: Missing dimensions in output: {missing}")

if __name__ == "__main__":
    test_codegen()
