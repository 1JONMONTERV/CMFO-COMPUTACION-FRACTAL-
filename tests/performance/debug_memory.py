
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("bindings/python"))

from cmfo.core.matrix import T7Matrix

def trace_instability():
    print("=== Tracing Memory Instability ===")
    
    # 1. Setup Single State
    # Use a random state that likely caused issues (large magnitude or specific phase)
    # Based on previous demo: random numbers + random imaginary
    state = np.random.rand(7) + 1j * np.random.rand(7)
    print(f"Initial Magnitude: {np.linalg.norm(state)}")
    
    mat = T7Matrix.identity()
    
    # 2. Step-by-Step Evolution
    v = state
    for i in range(1, 101):
        # Emulate the fallback logic: v = sin(v @ M.T)
        # Identity matrix -> v @ I = v
        # So v_new = sin(v_old)
        
        # In the demo it was: v = sin(v @ M_T)
        # Complex sine: sin(x + iy) = sin(x)cosh(y) + i cos(x)sinh(y)
        # sinh(y) grows EXPONENTIALLY with y.
        # If y (imaginary part) grows, magnitude explodes.
        
        v_next = mat.evolve_state(v, steps=1)
        
        mag = np.linalg.norm(v_next)
        print(f"Step {i}: Mag={mag:.4f} | RealRange=[{v_next.real.min():.2f}, {v_next.real.max():.2f}] | ImagRange=[{v_next.imag.min():.2f}, {v_next.imag.max():.2f}]")
        
        if not np.isfinite(mag):
            print(f"!!! CRASH/NAN at Step {i} !!!")
            break
            
        v = v_next

if __name__ == "__main__":
    trace_instability()
