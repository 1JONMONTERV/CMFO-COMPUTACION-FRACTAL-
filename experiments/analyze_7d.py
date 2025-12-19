"""
7D Hyper-Resolution Analysis Experiment
=======================================

Applies CMFO Hyper-Metrics (7D) to the mining dataset.
Goal: Measure resolution gain over standard metrics.
"""

import sys
import os
import json
import numpy as np

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bindings', 'python'))

from cmfo.core.fractal_algebra_1_1 import FractalUniverse1024
from cmfo.core.hyper_metrics import HyperMetrics

def analyze_7d():
    data_path = os.path.join(os.path.dirname(__file__), 'mining_dataset.json')
    with open(data_path) as f:
        dataset = json.load(f)

    print("="*60)
    print("   7D HYPER-RESOLUTION ANALYSIS")
    print("="*60)
    
    features = {}
    
    # 1. Compute Features
    for label, samples in dataset.items():
        vecs = []
        for s in samples:
            h_bytes = bytes.fromhex(s['header_hex'])
            padded = h_bytes + b'\x00' * (128 - len(h_bytes))
            u = FractalUniverse1024(padded)
            v = HyperMetrics.compute_7d(u)
            vecs.append(v)
        features[label] = np.array(vecs)

    # 2. Centroids and Variance
    centroids = {}
    print("\n[1] 7D Centroids per Class")
    print(f"  {'Class':<10} | {'D1(Ent)':<8} {'D2(Frac)':<8} {'D3(Chi)':<8} {'D4(Coh)':<8} {'D5(Top)':<8} {'D6(Phs)':<8} {'D7(Pot)':<8}")
    print("-" * 90)
    
    for label, vecs in features.items():
        mean = np.mean(vecs, axis=0)
        centroids[label] = mean
        # Print formatted
        print(f"  {label:<10} | {mean[0]:.4f}   {mean[1]:.4f}   {mean[2]:.4f}   {mean[3]:.4f}   {mean[4]:.4f}   {mean[5]:.4f}   {mean[6]:.4f}")

    # 3. Separability Analysis (Fishers? Or just Distance)
    # Distance from Random(0)
    print("\n[2] Distance from Random (Resolution Check)")
    ref = centroids['diff_0']
    
    for label in ['diff_8', 'diff_12', 'diff_16']:
        tgt = centroids[label]
        # Euclidean in 7D
        dist = np.linalg.norm(tgt - ref)
        
        # Compare to "Within Class Scatter" (Std Dev trace)
        # simplistic Fisher ratio proxy: Dist / (Std0 + Stdi)
        # Let's just print dist and variance.
        std_ref = np.mean(np.std(features['diff_0'], axis=0))
        std_tgt = np.mean(np.std(features[label], axis=0))
        
        fisher_proxy = dist / (std_ref + std_tgt)
        
        print(f"  {label:<10}: Dist={dist:.4f}  (Fisher Proxy={fisher_proxy:.4f})")
        
    # 4. Dimension Importance (Which D separates best?)
    print("\n[3] Dimension Contribution (Hard vs Random)")
    diff = np.abs(centroids['diff_16'] - centroids['diff_0'])
    dim_names = ["Entropy", "Fractal", "Chirality", "Coherence", "Topology", "Phase", "Potential"]
    
    # Sort by diff
    indices = np.argsort(-diff) # Descending
    for i in indices:
        print(f"  {dim_names[i]:<10}: Delta = {diff[i]:.4f}")

if __name__ == "__main__":
    analyze_7d()
