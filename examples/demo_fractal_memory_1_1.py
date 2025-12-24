"""
Demo: Fractal Memory 1.1 (Structural Indexing)
==============================================

Demonstrates the power of CMFO-FRACTAL-ALGEBRA 1.1 to index
data based on structural similarity (resonance) rather than bit-exactness.
"""

import sys
import os
import time
import secrets
import numpy as np

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bindings', 'python'))

from cmfo.memory.fractal_index import FractalIndex
from cmfo.core.fractal_algebra_1_1 import FractalUniverse1024

def generate_random_state() -> bytes:
    return secrets.token_bytes(128) # 1024 bits

def perturb_state(data: bytes, flips=5) -> bytes:
    """Flip random bits to create a 'structural relative'"""
    arr = bytearray(data)
    for _ in range(flips):
        idx = secrets.randbelow(len(arr))
        bit = secrets.randbelow(8)
        arr[idx] ^= (1 << bit)
    return bytes(arr)

def main():
    print("="*60)
    print("   FRACTAL MEMORY 1.1 DEMONSTRATION")
    print("   Structural Indexing of 1024-bit Universe")
    print("="*60)
    
    # Initialize Memory
    mem = FractalIndex()
    
    # 1. Populate Memory
    N_ITEMS = 100
    print(f"\n[1] Indexing {N_ITEMS} random universal states...")
    
    originals = []
    start = time.perf_counter()
    for i in range(N_ITEMS):
        data = generate_random_state()
        item_id = f"item_{i:04d}"
        mem.add(data, item_id)
        originals.append((item_id, data))
    
    elapsed = time.perf_counter() - start
    print(f"    Indexed in {elapsed:.4f}s ({N_ITEMS/elapsed:.0f} items/s)")
    stats = mem.get_stats()
    print(f"    Classes created: {stats['structural_classes']}")
    
    # 2. Structural Retrieval (Resonance)
    print(f"\n[2] Testing Resonance Retrieval (Perturbed inputs)...")
    
    hits = 0
    tests = 10
    
    for i in range(tests):
        # Pick a target
        target_id, target_data = originals[i]
        
        # Create a "noisy" version (structural neighbor)
        # Flip 10 bits (~1% noise). Hamming dist is 10.
        query_data = perturb_state(target_data, flips=10)
        
        # Search
        # We assume they fall in same bucket or d_MS is low
        # Note: bucket key is quantized Phivec. 
        # Small perturbations might shift key if near boundary.
        # Let's use `find_nearest` (Full scan) first to verify Metric Quality.
        
        results = mem.find_nearest(query_data, k=1)
        best_id, dist = results[0]
        
        match = (best_id == target_id)
        if match: hits += 1
        
        print(f"    Query structural relative of {target_id}:")
        print(f"      -> Found: {best_id} (Dist: {dist:.4f}) {'[MATCH]' if match else '[MISS]'}")

    print(f"    Accuracy: {hits}/{tests} ({hits/tests*100:.1f}%)")
    
    # 3. Anomaly Detection
    print(f"\n[3] Anomaly Detection (New alien state)...")
    alien = generate_random_state()
    # Should be far from everything else
    results = mem.find_nearest(alien, k=1)
    print(f"    Nearest neighbor dist: {results[0][1]:.4f}")
    if results[0][1] > 10.0: # Arbitrary threshold for random-random
        print("    -> Anomaly Detected (High Distance)")
    else:
        print("    -> No Anomaly (Dense space)")
        
    print("\n[SUCCESS] Fractal Memory 1.1 Operational.")

if __name__ == "__main__":
    main()
