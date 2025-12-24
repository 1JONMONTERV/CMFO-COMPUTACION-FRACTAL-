import time
import numpy as np
import sys
import os

# Ensure we can load local package
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))
import cmfo
from cmfo.core.matrix import T7Matrix

def benchmark_matrix_apply():
    print("=== CMFO High-Performance Benchmark ===")
    print("Task: Matrix-Vector Multiplication (100,000 ops)")
    
    # Setup
    N_OPS = 100000
    vec = np.random.rand(7)
    mat_np = np.eye(7)
    
    # 1. Numpy Baseline
    t0 = time.time()
    v = vec.copy()
    for _ in range(N_OPS):
        v = mat_np @ v
    t_numpy = time.time() - t0
    print(f"Numpy (Python): {t_numpy:.4f} s")
    
    # 2. C++ Native Engine
    try:
        mat_native = T7Matrix.identity()
        t0 = time.time()
        # Note: calling 'apply' from python still has overhead.
        # Ideally we loop internally in C++. But let's test single-op overhead vs Numpy.
        v_native = vec.copy()
        for _ in range(N_OPS):
            v_native = mat_native.apply(v_native)
        t_native = time.time() - t0
        print(f"C++ Engine (via ctypes): {t_native:.4f} s")
        
        speedup = t_numpy / t_native
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.0:
            print("RESULT: C++ Engine is FASTER.")
        else:
            print("RESULT: C++ Engine overhead is simpler than Numpy BLAS for small 7x7?")
            
    except Exception as e:
        print(f"C++ Engine Failed: {e}")
        print("Did you compile the extension?")

if __name__ == "__main__":
    benchmark_matrix_apply()
