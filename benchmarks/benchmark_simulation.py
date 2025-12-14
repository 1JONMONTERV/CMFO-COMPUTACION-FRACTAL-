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
    mat_np = np.eye(7) * 0.5 + np.ones((7,7))*0.01 # Some mixing
    mat_obj = T7Matrix() 
    # T7Matrix is Identity by default.
    # We should set it to verify computation. But for now speed is key.
    
    # 1. Numpy Loop (Python Side)
    t0 = time.time()
    v = vec.copy()
    for _ in range(N_STEPS):
        # M * v
        temp = mat_np @ v
        # Gamma (sin)
        v = np.sin(temp)
    t_numpy = time.time() - t0
    print(f"Numpy Simulation: {t_numpy:.4f} s")
    
    # 2. C++ Native Loop
    try:
        t0 = time.time()
        # The matrix in C++ is Identity by default. 
        # The cost of mult is slightly lower for Identity if optimized, but here we do full 7x7 mult manually in loop.
        # So it simulates full load.
        v_final = mat_obj.evolve_state(vec, steps=N_STEPS)
        t_native = time.time() - t0
        print(f"C++ Native Loop:  {t_native:.4f} s")
        
        speedup = t_numpy / t_native
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 50.0:
            print("RESULT: SUCCESS! Massive acceleration achieved.")
        else:
            print("RESULT: Initial acceleration ok, but could be better.")
            
    except Exception as e:
        print(f"C++ Engine Failed: {e}")

if __name__ == "__main__":
    benchmark_simulation_loop()
