"""
benchmarks/cmfo_vs_transformer.py
---------------------------------
Simulates the computational cost of Self-Attention (Quadratic) vs 
CMFO Fractal Projection (Linear/Constant).

This is a synthetic benchmark to demonstrate algorithmic scaling.
"""

import time
import numpy as np

def simulate_transformer_attention(seq_len, dim):
    """
    Simulate O(N^2) complexity of Attention Matrix Multiplications.
    """
    # Create fake Q, K matrices
    Q = np.random.rand(seq_len, dim)
    K = np.random.rand(seq_len, dim)
    
    start = time.time()
    # PURE MATH SIMULATION: Q * K^T
    # This is the heavy step in Transformers
    attn_scores = np.dot(Q, K.T)
    # Simulate Softmax
    attn_probs = np.exp(attn_scores) / np.sum(np.exp(attn_scores), axis=1, keepdims=True)
    end = time.time()
    
    return end - start

def simulate_cmfo_projection(seq_len, dim):
    """
    Simulate O(N) complexity of CMFO Fractal Absorption.
    Actually, it's O(N) to ingest, but state size is O(1) (7 dimensions).
    """
    # Create fake tokens
    tokens = np.random.rand(seq_len, 7) # CMFO operates on 7D projection
    
    start = time.time()
    # PURE MATH SIMULATION: Sequential Absorption
    # State is fixed size (7,)
    state = np.zeros(7)
    PHI = 1.618
    
    # We can vectorize this simulation for Python speed, 
    # but logically it is a scan.
    # To be fair to the "Simulated Logic", we run the loop 
    # as CMFO relies on sequential dependency (like RNNs but mathematical)
    for t in tokens:
        # T7 Operator: (State * Input + PHI) / (1 + PHI)
        state = (state * t + PHI) / (1 + PHI)
        
    end = time.time()
    
    return end - start

def main():
    print("--- CMFO vs Transformer Scaling Benchmark ---")
    print(f"{'Seq Length':<15} | {'Transformer (s)':<15} | {'CMFO (s)':<15} | {'Speedup':<10}")
    print("-" * 65)
    
    for n in [100, 1000, 2000, 5000]:
        dim = 64 # Standard head dimension
        
        t_trans = simulate_transformer_attention(n, dim)
        t_cmfo = simulate_cmfo_projection(n, dim)
        
        speedup = t_trans / t_cmfo if t_cmfo > 0 else 9999
        
        print(f"{n:<15} | {t_trans:.6f}          | {t_cmfo:.6f}          | {speedup:.1f}x")

if __name__ == "__main__":
    main()
