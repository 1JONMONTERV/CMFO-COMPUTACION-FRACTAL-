"""
Deep Analysis of Mining Topology (Algebra 1.1)
==============================================

Analyzes the structural properties of SHA-256d states at varying difficulty levels.
Uses CMFO-FRACTAL-ALGEBRA 1.1 to map headers to Phi_90 space.

Hypothesis:
1. Golden solutions (high difficulty) form a distinct cluster in Phi_90 space.
2. There exists a structural gradient (d_MS) towards higher difficulty.
"""

import sys
import os
import json
import numpy as np
import math

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bindings', 'python'))

from cmfo.core.fractal_algebra_1_1 import FractalUniverse1024, Metrics

def analyze_dataset():
    data_path = os.path.join(os.path.dirname(__file__), 'mining_dataset.json')
    with open(data_path) as f:
        dataset = json.load(f)
        
    print("="*60)
    print("   MINING TOPOLOGY ANALYSIS (CMFO 1.1)")
    print("="*60)
    
    # Storage for features per class
    # Class -> List of Phi_90 vectors (size 90)
    features = {} 
    universes = {}
    
    for label, samples in dataset.items():
        feats = []
        unis = []
        for s in samples:
            # We assume header_hex is the 80-byte input
            # BUT wait, the header is 80 bytes (640 bits).
            # FractalUniverse1024 requires 1024 bits.
            # We must PAD the header to 1024 bits (standard SHA block padding).
            # SHA-256 padding for 640 bits:
            # 640 bits = 80 bytes.
            # Block size 512 bits. 640 > 512. It takes 2 blocks.
            # The "State" we analyze... is it the INPUT or the HASH?
            # User request: "Analizar la mineria".
            # The "Solution" is the Nonce (Input).
            # The "Target" is the Hash (Output).
            # The Algebra can analyze both.
            # If we analyze the Input (Header), we look for patterns in nonces.
            # If we analyze the Hash, we look for patterns in the Output Space.
            # Mining is searching Input Space to hit Output Target.
            # Let's analyze the INPUT structural invariants. (Do 'winning' headers have structure?)
            # And analyze the OUTPUT structural invariants. (Do 'winning' hashes share Phi90 structure?)
            # Output structure is trivial: Leading zeros -> Low entropy in MSB.
            # Input structure is non-trivial.
            
            # Let's analyze INPUTS (Header with correct nonce).
            h_bytes = bytes.fromhex(s['header_hex'])
            # Pad to 128 bytes (1024 bits)
            padded = h_bytes + b'\x00' * (128 - len(h_bytes))
            
            u = FractalUniverse1024(padded)
            phi = Metrics.phi_90(u)
            
            feats.append(phi)
            unis.append(u)
            
        features[label] = np.array(feats)
        universes[label] = unis
        
    # 1. Centroid Analysis (Phi 90)
    print("\n[1] Centroid Analysis (Input Space Structure)")
    centroids = {}
    for label, feats in features.items():
        mean_vec = np.mean(feats, axis=0)
        std_vec = np.std(feats, axis=0)
        centroids[label] = mean_vec
        print(f"  {label:<10}: Mean Phi Norm = {np.linalg.norm(mean_vec):.4f}, Std Norm = {np.linalg.norm(std_vec):.4f}")
        
    # 2. Inter-Class Distances (d_MS centroids)
    # Are the "Easy" solutions closer to "Hard" solutions than "Random"?
    print("\n[2] Structural Gradient (Euclidean in Phi Space)")
    ref = centroids['diff_0']
    for label in ['diff_8', 'diff_12', 'diff_16']:
        tgt = centroids[label]
        dist = np.linalg.norm(tgt - ref)
        print(f"  Dist(Diff_0 -> {label}): {dist:.4f}")
        
    # 3. Variance Analysis (Focusing)
    # Does the solution space "shrink" (lower variance) as difficulty increases?
    # This implies a "Funnel" topology.
    print("\n[3] Manifold Focusing (Variance Analysis)")
    for label, feats in features.items():
        # Trace of Covariance matrix (Total Variance)
        # Approximate: Sum of variances of components
        total_var = np.sum(np.var(feats, axis=0))
        print(f"  {label:<10}: Total Variance = {total_var:.4f}")
        
    # 4. Hash Analysis (Output Space) - Sanity Check
    # Winning hashes SHOULD have very distinct Phi90 because of zeros.
    print("\n[4] Output Space Analysis (The Target)")
    hash_features = {}
    for label, samples in dataset.items():
        h_feats = []
        for s in samples:
            h_bytes = bytes.fromhex(s['hash_hex'])
            padded = h_bytes + b'\x00' * (128 - len(h_bytes))
            u_h = FractalUniverse1024(padded)
            h_feats.append(Metrics.phi_90(u_h))
        hash_features[label] = np.array(h_feats)
        
    for label, h_feats in hash_features.items():
        mean_h = np.mean(h_feats, axis=0)
        print(f"  {label:<10} Output Phi Norm: {np.linalg.norm(mean_h):.4f}")

if __name__ == "__main__":
    analyze_dataset()
