#!/usr/bin/env python3
"""
Fractal-Guided Mining Demo
===========================

Demonstrates mining using fractal geometry instead of brute-force.
Uses the Torus attractor dynamics to guide the search.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, '..', 'bindings', 'python')
sys.path.append(bindings_path)

from cmfo.mining.fractal_miner import FractalMiner
from cmfo.mining.optimized_distiller import OptimizedDistiller

def main():
    print("=" * 60)
    print("   CMFO FRACTAL-GUIDED MINING")
    print("   Strategy: Geometric Gradient Descent")
    print("=" * 60)
    
    # Initialize
    miner = FractalMiner(torus_size=1024)
    distiller = OptimizedDistiller()
    
    # Create header
    base_header = distiller.create_base_header(
        version=2,
        prev_hash=b'\xaa' * 32,
        merkle_root=b'\xff' * 32
    )
    
    print(f"\n[Header] {base_header.hex()[:32]}...")
    
    # Mine using fractal guidance
    DIFFICULTY = 4  # 4 zeros = more achievable for demo
    MAX_ATTEMPTS = 50000
    
    print(f"\n[Config]")
    print(f"  Difficulty: {DIFFICULTY} zeros")
    print(f"  Max Attempts: {MAX_ATTEMPTS:,}")
    print(f"  Method: Phi-spiral + Gradient descent")
    
    nonce, hash_result = miner.fractal_mine(
        base_header=base_header,
        target_difficulty=DIFFICULTY,
        max_attempts=MAX_ATTEMPTS
    )
    
    if nonce is not None:
        print(f"\n{'='*60}")
        print("SUCCESS - Solution Found!")
        print('='*60)
        print(f"  Nonce: {nonce:,}")
        print(f"  Hash: {hash_result}")
        print(f"\n  Fractal mining VALIDATED ✓")
    else:
        print(f"\n  Try increasing max_attempts or lowering difficulty")

if __name__ == "__main__":
    main()
