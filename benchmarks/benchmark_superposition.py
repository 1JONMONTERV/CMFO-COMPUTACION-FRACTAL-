import time
import numpy as np
import sys
import os

# Ensure we can load local package
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))
import cmfo
from cmfo.core.matrix import T7Matrix

def benchmark_superposition():
    print("=== CMFO Superposition Benchmark (Multi-Node) ===")
    print("Task: Evolve 10,000 Concurrent Fractal Universes (States) for 100 Steps")
    
    # Setup
    N_NODES = 10000
    N_STEPS = 100
    
    # Create N random states
    states = np.random.rand(N_NODES, 7) + 1j * np.random.rand(N_NODES, 7)
    mat_obj = T7Matrix() 
    
    print(f"Nodes: {N_NODES}")
    print(f"Steps: {N_STEPS}")
    print(f"Total Operations: {N_NODES * N_STEPS} matrix evolutions")
    
    # 1. Python Loop (Simulating pure python overhead)
    # Note: Using Numpy broadcasting is fast, but let's see how fast "C++ Evolve" is vs Python loop approach
    # Ideally standard Python user might loop. But even compared to Numpy optimized batch, 
    # C++ might win due to avoiding Python interpreter overhead for iterator.
    
    print("\n--- Method A: Python Loop (Standard) ---")
    t0 = time.time()
    # To be fair to Python, we won't loop 10,000 times explicitly if we can help it, 
    # but simulating "Independent Agents" usually implies a loop or map.
    # Let's try a realistic batch processing using Numpy broadcasting (Best case for Python)
    
    # Numpy Broadcasting Implementation of Evolve
    # v_new = sin(M @ v)
    # v is (N, 7), M is (7, 7). 
    # we need (N, 7) @ (7, 7).T? No, v @ M.T
    
    batch_py = states.copy()
    mat_np = np.eye(7) # Identity
    
    for _ in range(N_STEPS):
        # M * v.T -> v @ M.T
        temp = batch_py @ mat_np
        batch_py = np.sin(temp)
        
    t_python = time.time() - t0
    print(f"Python (Numpy Batch): {t_python:.4f} s")
    
    # 2. C++ Native Batch (Superposition Engine)
    print("\n--- Method B: CMFO C++ Superposition Engine ---")
    try:
        t0 = time.time()
        # This calls one C function that loops internally in C++
        # Zero python overhead inside the loop
        res_cpp = mat_obj.evolve_batch(states, steps=N_STEPS)
        t_cpp = time.time() - t0
        print(f"C++ Native Batch:     {t_cpp:.4f} s")
        
        speedup = t_python / t_cpp
        print(f"\nSpeedup: {speedup:.2f}x")
        
        if speedup > 1.0:
             print("RESULT: C++ Engine beats optimized Numpy Batch!")
             print("Note: This proves C++ is efficient even against BLAS for this specific Gamma-Loop.")
        else:
             print("RESULT: Numpy is competitive (BLAS is strong).")
            
    except Exception as e:
        print(f"C++ Engine Failed: {e}")

if __name__ == "__main__":
    benchmark_superposition()
