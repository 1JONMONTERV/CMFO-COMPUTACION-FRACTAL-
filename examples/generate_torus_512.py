#!/usr/bin/env python3
"""
Generate 512x512 Fractal Torus
================================

Creates a manageable-sized fractal space for solution mapping.
512^2 = 262,144 states (much faster than 2^32).
"""

import sys
import os
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, '..', 'bindings', 'python')
sys.path.append(bindings_path)

from cmfo.topology.fractal_torus import FractalTorus

def main():
    print("=" * 60)
    print("   CMFO FRACTAL TORUS 512x512")
    print("   Resolution: 262,144 states")
    print("=" * 60)
    
    SIZE = 512
    STEPS = 50
    
    print(f"\n[Init] Creating {SIZE}x{SIZE} Torus...")
    ft = FractalTorus(size=SIZE, kernel_size=7)
    
    print(f"[Evolution] Running {STEPS} steps...")
    start = time.time()
    
    for i in range(STEPS):
        ft.step()
        if (i + 1) % 10 == 0:
            metrics = ft.measure_geometry()
            print(f"  Step {i+1:02d} | Tensor: {metrics['tensor_trace_mean']:.6f} | "
                  f"Entropy: {metrics['entropy']:.6f}")
    
    elapsed = time.time() - start
    
    # Final metrics
    final = ft.measure_geometry()
    
    print(f"\n{'='*60}")
    print("FINAL STATE")
    print('='*60)
    print(f"  Size: {SIZE} x {SIZE}")
    print(f"  Total States: {SIZE*SIZE:,}")
    print(f"  Evolution Time: {elapsed:.2f}s")
    print(f"  Tensor Trace: {final['tensor_trace_mean']:.9f}")
    print(f"  Entropy: {final['entropy']:.9f}")
    print(f"  Mean Angle: {final['mean_angle']:.9f} rad")
    print(f"  Attractor: {final['in_attractor']}")
    
    # Save state
    output_file = os.path.join(current_dir, 'fractal_torus_512.npy')
    np.save(output_file, ft.grid)
    print(f"\n[Save] State saved to: {output_file}")
    print(f"  File size: {os.path.getsize(output_file)/1024:.2f} KB")
    
    print(f"\n  Status: READY FOR SOLUTION MAPPING ✓")

if __name__ == "__main__":
    main()
