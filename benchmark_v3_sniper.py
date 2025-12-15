
"""
CMFO v3.0 "The Sniper" Benchmark
================================
Demonstrates the performance advantage of Fused Kernels (v3.0) vs Generic Ops (PyTorch-style).

We simulate the physics formula:
    z = (v * h + PHI) / (1 + h)

Method A (Legacy/PyTorch): 3 separate passes over memory, creating intermediate buffers.
Method B (CMFO Sniper): 1 single fused pass, zero intermediate memory.
"""

import time
import math
import random

# Constants
PHI = 1.6180339887
DIM = 7
BATCH_SIZE = 50000 # Sizeable batch to show impact
ITERATIONS = 50

def setup_data():
    print(f"Generating data: {BATCH_SIZE} vectors of dim {DIM}...")
    v = [[random.random() for _ in range(DIM)] for _ in range(BATCH_SIZE)]
    h = [[random.random() for _ in range(DIM)] for _ in range(BATCH_SIZE)]
    return v, h

def pytorch_style_execution(v_batch, h_batch):
    """
    Simulates:
    1. tmp1 = v * h
    2. tmp2 = 1 + h
    3. tmp3 = tmp1 + PHI
    4. out = tmp3 / tmp2
    
    Creates 4 FULL intermediate lists (memory traffic).
    """
    # 1. Mul
    tmp1 = [[v_val * h_val for v_val, h_val in zip(v_row, h_row)] 
            for v_row, h_row in zip(v_batch, h_batch)]
            
    # 2. Add Scalar (1+h)
    tmp2 = [[1.0 + h_val for h_val in h_row] for h_row in h_batch]
    
    # 3. Add Scalar (tmp1 + PHI)
    tmp3 = [[val + PHI for val in row] for row in tmp1]
    
    # 4. Div
    out = [[n / d for n, d in zip(num, den)] for num, den in zip(tmp3, tmp2)]
    return out

def cmfo_sniper_execution(v_batch, h_batch):
    """
    Simulates CMFO v3.0 Fused Kernel.
    One single pass expressible as:
    out[i] = (v[i]*h[i] + PHI) / (1 + h[i])
    
    Zero intermediate lists.
    """
    out = []
    # In C++, this loop is fully unrolled by the compiler for 7D
    for v_row, h_row in zip(v_batch, h_batch):
        # FUSED OPERATION LINE
        row = [(v * h + PHI) / (1.0 + h) for v, h in zip(v_row, h_row)]
        out.append(row)
    return out

def run_benchmark():
    v, h = setup_data()
    
    print(f"\n--- Running Benchmark ({ITERATIONS} iterations) ---")
    
    # Method A
    start = time.time()
    for _ in range(ITERATIONS):
        _ = pytorch_style_execution(v, h)
    end = time.time()
    time_a = end - start
    print(f"Creating intermediate buffers (PyTorch-style): {time_a:.4f}s")
    
    # Method B
    start = time.time()
    for _ in range(ITERATIONS):
        _ = cmfo_sniper_execution(v, h)
    end = time.time()
    time_b = end - start
    print(f"Fused Kernel (CMFO Sniper):                {time_b:.4f}s")
    
    # Analysis
    ratio = time_a / time_b
    print("\n--- RESULTS ---")
    print(f"Speedup: {ratio:.2f}x")
    print("Conclusion: Fusing operations reduces memory allocation and iteration overhead.")
    print("In GPU, this difference is magnified due to VRAM latency savings.")

if __name__ == "__main__":
    run_benchmark()
