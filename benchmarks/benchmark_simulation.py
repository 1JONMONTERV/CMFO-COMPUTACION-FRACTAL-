import time
import numpy as np
import sys
import os

# Ensure we can load local package
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))
import cmfo
from cmfo.core.matrix import T7Matrix

def benchmark_simulation_loop():
    print("=== CMFO Simulation Loop Benchmark ===")
    print("Task: Evolve State v_{t+1} = sin(M * v_t) for 100,000 steps")
    
    # Setup
    N_STEPS = 100000
    vec = np.random.rand(7)
    
    # 0. Check Native Availability
    try:
        from cmfo.core.native_lib import NativeLib
        lib = NativeLib.get()
        if lib:
            print("Native Acceleration: ENABLED ✅")
        else:
            print("Native Acceleration: DISABLED ❌ (Library not found)")
    except Exception as e:
        print(f"Native Acceleration: ERROR ❌ ({e})")

    # 1. Numpy Loop (Python Side)
    # Warmup
    print("Running NumPy warmup...")
    mat_np = np.eye(7)
    v = vec.copy()
    for _ in range(100): 
        v = np.sin(mat_np @ v)
        
    print("Running NumPy benchmark...")
    t0 = time.time()
    v = vec.copy()
    
    # Python Loop Overhead + Numpy Overhead
    # mat_np is identity, so optimized BLAS might be fast, but Python loop dominates
    for _ in range(N_STEPS):
        temp = mat_np @ v
        v = np.sin(temp)
    t_numpy = time.time() - t0
    print(f"Numpy Simulation: {t_numpy:.4f} s")
    
    # 2. C++ Native Loop
    try:
        mat_obj = T7Matrix() # Uses native if available
        
        # Warmup
        print("Running Native warmup...")
        mat_obj.evolve_state(vec, steps=100)
        
        print("Running Native benchmark...")
        t0 = time.time()
        # The entire loop is inside C++
        v_final = mat_obj.evolve_state(vec, steps=N_STEPS)
        t_native = time.time() - t0
        print(f"C++ Native Loop:  {t_native:.4f} s")
        
        speedup = t_numpy / t_native
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 50.0:
            print("RESULT: SUCCESS! Massive acceleration achieved. 🚀")
        elif speedup > 10.0:
            print("RESULT: Good acceleration, but check optimizations.")
        else:
            print("RESULT: Initial acceleration ok, but could be better.")
            
    except Exception as e:
        print(f"C++ Engine Failed/Not Used: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    benchmark_simulation_loop()
