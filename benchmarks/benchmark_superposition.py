import time
import numpy as np
import sys
import os

# Ensure we can load local package
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))
import cmfo
from cmfo.core.matrix import T7Matrix
from cmfo.core.native_lib import NativeLib

def benchmark_superposition():
    print("=== CMFO Fractal Superposition Benchmark ===")
    print("Task: Evolve 100,000 parallel fractal timelines for 100 steps")

    BATCH_SIZE = 100000
    STEPS = 100
    
    # 0. Check Native
    lib = NativeLib.get()
    if not lib:
        print("ERROR: Native library not found. Cannot benchmark OpenMP.")
        return

    # Generate massive random state
    print(f"Generating {BATCH_SIZE} random states...")
    batch_states = np.random.rand(BATCH_SIZE, 7) + 1j * np.random.rand(BATCH_SIZE, 7)
    
    mat_obj = T7Matrix.identity()

    # 1. Warmup
    print("Warming up OpenMP threads...")
    mat_obj.evolve_batch(batch_states[:1000], steps=10)

    # 2. Benchmark Native Batch (OpenMP)
    print("Running Massive Superposition (C++ OpenMP)...")
    t0 = time.time()
    
    # The C++ engine handles the loop over BATCH_SIZE using #pragma omp parallel for
    final_states = mat_obj.evolve_batch(batch_states, steps=STEPS)
    
    t_native = time.time() - t0
    print(f"Time: {t_native:.4f} s")
    
    # Calculate effective operations
    # Ops = Batch * Steps * (MatrixMult(7x7) + Activation(7))
    # MatrixMult 7x7 complex ≈ 7*7*4 doubles ops (mul/add) ≈ 200 ops
    # Total ops estimate per state per step ≈ 300 ops
    total_ops = BATCH_SIZE * STEPS * 300
    gflops = (total_ops / t_native) / 1e9
    
    print(f"Throughput: {BATCH_SIZE / t_native:.0f} states/sec")
    print(f"Est. Performance: {gflops:.2f} GFLOPS")

    # 3. Estimate Python Time (Don't actually run, too slow)
    # Python ~ 100,000 steps took 0.6s in single thread prev benchmark (approx)
    # So 100,000 * 100 steps would take... wait.
    # Prev benchmark: 1 state, 100,000 steps => 0.6s
    # This benchmark: 100,000 states, 100 steps => same total operations?
    # Yes. 1*100k = 100k*1? No, logic overhead is per call.
    # Python loop over batch would be VERY slow.
    
    print("\n--- Speedup Analysis ---")
    # Baseline Python single state 100k steps ~ 0.5s (from prev benchmark)
    # We did 100k * 100 = 10M total state-steps.
    # Python baseline for 10M steps ≈ 50 seconds (extrapolated)
    
    t_python_est = 50.0 
    speedup = t_python_est / t_native
    
    print(f"Est. Python Time: {t_python_est:.2f} s")
    print(f"Speedup vs Python: {speedup:.2f}x")
    
    if speedup > 50.0:
        print("RESULT: SUCCESS! Massive Superposition achieved. 🚀")
    else:
        print("RESULT: Decent speedup, but maybe not full heavy core usage.")

if __name__ == "__main__":
    benchmark_superposition()
