"""
Positional Analysis Experiment
==============================

Search for the 'Natural Coordinate System' that minimizes structural variance
of Golden Solutions in the 7D Phase space.
"""

import sys
import os
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bindings', 'python'))

from cmfo.core.fractal_algebra_1_1 import FractalUniverse1024
from cmfo.core.hyper_metrics import HyperMetrics
from cmfo.core.positional import PositionalAlgebra

def analyze_positional():
    print("="*60)
    print("   POSITIONAL VALUE ANALYSIS")
    print("   Searching for the Natural Geometric Frame")
    print("="*60)
    
    # Load Signal
    data_path = os.path.join(os.path.dirname(__file__), 'mining_dataset.json')
    with open(data_path) as f:
        dataset = json.load(f)
    
    # Golden samples (Diff 12 & 16)
    signal_hex = [x['header_hex'] for x in dataset['diff_12']] + \
                 [x['header_hex'] for x in dataset['diff_16']]
    
    signal = []
    for hx in signal_hex:
        b = bytes.fromhex(hx)
        signal.append(b + b'\x00'*(128-len(b)))

    print(f"Loaded {len(signal)} Golden Samples.")

    # Define Candidates
    transforms = {
        "Identity": PositionalAlgebra.delta_flat(shift=0),
        "Linear (k=1)": PositionalAlgebra.delta_linear(slope=1),
        "Balanced (Anti-Sym)": PositionalAlgebra.delta_balanced(),
        "Octagonal (Word)": PositionalAlgebra.delta_octagonal(),
        "Reverse Linear": PositionalAlgebra.delta_linear(slope=-1),
        "Quadratic?": (np.arange(256)**2 % 16).astype(int)
    }

    print(f"\n{'Transform':<20} | {'Mean Phase':<10} | {'Std Dev':<10} | {'Improvement'}")
    print("-" * 65)

    base_std = 0.0

    for name, delta in transforms.items():
        phases = []
        for data in signal:
            u_raw = FractalUniverse1024(data)
            # Apply Transform
            u_trans = PositionalAlgebra.apply(u_raw, delta)
            
            # Compute Metric (D6 Phase)
            v = HyperMetrics.compute_7d(u_trans)
            phases.append(v[5])
            
        mean_p = np.mean(phases)
        std_p = np.std(phases)
        
        if name == "Identity":
            base_std = std_p
            improv = 0.0
        else:
            improv = (base_std - std_p) / base_std * 100
            
        print(f"{name:<20} | {mean_p:.4f}     | {std_p:.4f}     | {improv:+.1f}%")

    print("\n[Interpretation]")
    print("Positive Improvement (+) means the transform TIGHTENS the cluster (Focusing).")
    print("Negative Improvement (-) means the transform SCATTERS the cluster (Blurring).")

if __name__ == "__main__":
    analyze_positional()
