"""
Mining Optimization Evaluation
==============================

Tests the viability of a "Phase-Guided" mining filter.
Filter Criteria (Derived from Hyper-Resolution Analysis):
1. Phase (D6) in [0.85, 0.96] (Targeting the 0.938 anomaly with tolerance)
2. Phi Norm (D1+D2 proxy) > 7.0 (Targeting the 7.15 shift)

Goal: High Rejection of Randoms, High Retention of Gold.
"""

import sys
import os
import json
import numpy as np
import random
import struct

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bindings', 'python'))

from cmfo.core.fractal_algebra_1_1 import FractalUniverse1024
from cmfo.core.hyper_metrics import HyperMetrics

def generate_random_headers(count=1000):
    headers = []
    # Realistic Header Structure
    ver = struct.pack("<I", 1)
    prev = b'\x00' * 32
    root = b'\x00' * 32 # Constant root for noise test? Or random root?
    # In mining sampler we varied root. Let's vary root to match dataset variance.
    ts = struct.pack("<I", 123456789)
    bits = b'\xff\xff\xff\xff'
    
    for _ in range(count):
        nonce = os.urandom(4)
        root = os.urandom(32) # Random root to mimic independent blocks
        h = ver + prev + root + ts + bits + nonce
        headers.append(h)
    return headers

def is_candidate(u: FractalUniverse1024) -> bool:
    """The 7D Pre-Filter"""
    # Create lightweight metrics first if possible?
    # For this test, we compute full 7D.
    v = HyperMetrics.compute_7d(u)
    
    # D6: Octagonal Phase
    phase = v[5]
    
    # D1+D2+... : Fractal Complexity (Norm of first 3 dims?)
    # Previous report said "Mean Phi Norm" (of all 90?) 
    # HyperMetrics compute_7d returns 7 dims.
    # The "Phi Norm" in previous report was L2 of Phi90 (90 dims).
    # Here we have 7 dims.
    # Let's use D1 (Entropy) and D6 (Phase) as key discriminators.
    
    # Criteria 1: Phase Rotation (Signal ~ 0.9, Noise ~ 0.4)
    # Filter: Keep High Phase
    if phase < 0.7:
        return False
        
    # Criteria 2: Entropy (Signal ~ 0.16, Noise ~ 0.53)
    # Filter: Keep Low Entropy (Structured)
    # Note: Report said Signal > Random(0.14), but Noise(0.53) is high due to random root.
    entr = v[0]
    if entr > 0.25:
        return False
        
    return True

def evaluate():
    print("="*60)
    print("   MINING OPTIMIZATION EVALUATION")
    print("   Strategy: Phase-Guided Filtering")
    print("="*60)
    
    # 1. Load Signal (Golden Samples)
    data_path = os.path.join(os.path.dirname(__file__), 'mining_dataset.json')
    with open(data_path) as f:
        dataset = json.load(f)
        
    # Combine Medium and Hard (diff_12, diff_16)
    signal_hex = [x['header_hex'] for x in dataset['diff_12']] + \
                 [x['header_hex'] for x in dataset['diff_16']]
    
    signal = []
    for hx in signal_hex:
        b = bytes.fromhex(hx)
        signal.append(b + b'\x00'*(128-len(b)))
        
    print(f"[1] Signal Dataset (Golden): {len(signal)} samples")
    
    # 2. Generate Noise (Random Samples)
    print(f"[2] Noise Dataset (Random): Generating 1000 samples...")
    noise_raw = generate_random_headers(1000)
    noise = [b + b'\x00'*(128-len(b)) for b in noise_raw]
    
    # 3. Assess Filter & Calibrate
    print("\n[3] Calibrating Thresholds (Inspect Distributions)...")
    
    sig_phases = []
    sig_ents = []
    for data in signal:
        u = FractalUniverse1024(data)
        v = HyperMetrics.compute_7d(u)
        sig_phases.append(v[5])
        sig_ents.append(v[0])
        
    noise_phases = []
    noise_ents = []
    # Sample subset of noise for speed
    for data in noise[:100]:
        u = FractalUniverse1024(data)
        v = HyperMetrics.compute_7d(u)
        noise_phases.append(v[5])
        noise_ents.append(v[0])
        
    print(f"  Signal Phase: Mean={np.mean(sig_phases):.3f}, Min={np.min(sig_phases):.3f}, Max={np.max(sig_phases):.3f}")
    print(f"  Signal Ent  : Mean={np.mean(sig_ents):.3f}, Min={np.min(sig_ents):.3f}, Max={np.max(sig_ents):.3f}")
    
    print(f"  Noise Phase : Mean={np.mean(noise_phases):.3f}, Min={np.min(noise_phases):.3f}, Max={np.max(noise_phases):.3f}")
    print(f"  Noise Ent   : Mean={np.mean(noise_ents):.3f}, Min={np.min(noise_ents):.3f}, Max={np.max(noise_ents):.3f}")

    print("\n[4] Running Simulation with Calibrated Logic...")
    
    # Check Signal Retention (True Positives)
    tp = 0
    for data in signal:
        u = FractalUniverse1024(data)
        if is_candidate(u):
            tp += 1
            
    recall = tp / len(signal)
    
    # Check Noise Rejection (True Negatives)
    tn = 0
    for data in noise:
        u = FractalUniverse1024(data)
        if not is_candidate(u):
            tn += 1
            
    rejection = tn / len(noise)
    fpr = 1.0 - rejection # False Positive Rate
    
    print("\n" + "="*30)
    print("   RESULTS")
    print("="*30)
    print(f"Recall (Golden retained) : {recall:.1%} ({tp}/{len(signal)})")
    print(f"Rejection (Random dropped): {rejection:.1%} ({tn}/{len(noise)})")
    print(f"False Positive Rate       : {fpr:.1%}")
    
    # Efficiency
    if fpr > 0:
        enrichment = (recall / fpr)
    else:
        enrichment = 999.0
        
    print(f"\nOptimization Factor (Enrichment): {enrichment:.2f}x")
    
    if rejection > 0.5 and recall > 0.9:
        print("\n[VERDICT] VIABLE STRATEGY. Reduces hashing load significantly.")
    else:
        print("\n[VERDICT] REQUIRES TUNING. Trade-off not optimal.")

if __name__ == "__main__":
    evaluate()
