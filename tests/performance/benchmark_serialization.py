
import time
import sys
import os
import array
import itertools
import ctypes

# Benchmark strictly the "Overhead" of the bridge (Marshalling)
# We replicate the logic inside gpu.py's wrapper to measure it isolated.

def benchmark_marshalling():
    print("==================================================")
    print("   [+] MARSHALLING OVERHEAD BENCHMARK [+]")
    print("==================================================")
    
    BATCH = 5000
    DIM = 1024
    
    # 1. SETUP DATA
    print("[1] Generating Data...")
    list_data = [ [0.1]*DIM for _ in range(BATCH) ]
    array_data = array.array('f', [0.1]*(BATCH*DIM))
    
    print(f"    Batch: {BATCH}, Dim: {DIM}")
    
    # 2. LIST STRATEGY (Naive)
    print("\n[2] Benchmarking LIST Strategy (Copy & Convert)...")
    t0 = time.time()
    
    # The costly steps from gpu.py:
    # 1. Chain
    flat_input = list(itertools.chain(*list_data))
    # 2. Ctypes Array Creation (Copy)
    FloatArray = ctypes.c_float * len(flat_input)
    c_in = FloatArray(*flat_input)
    
    t1 = time.time()
    list_time = t1 - t0
    print(f"    Time: {list_time:.6f} sec")
    
    # 3. ARRAY STRATEGY (optimized)
    print("\n[3] Benchmarking ARRAY Strategy (Zero Copy)...")
    t2 = time.time()
    
    # The optimized steps:
    # 1. Infer types (instant)
    # 2. From Buffer (Pointer Math)
    c_in_opt = (ctypes.c_float * len(array_data)).from_buffer(array_data)
    
    t3 = time.time()
    array_time = t3 - t2
    print(f"    Time: {array_time:.6f} sec")
    
    # 4. RESULT
    speedup = list_time / array_time if array_time > 0 else 9999
    print("\n==================================================")
    print(f"   [+] SPEEDUP: {speedup:.1f}x")
    print("==================================================")

if __name__ == "__main__":
    benchmark_marshalling()
