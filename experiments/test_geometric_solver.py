"""
Geometric Nonce Solver - Proof of Concept
=========================================

Tests the hypothesis that we can find valid nonces by solving
the Quadratic Phase Constraint rather than brute-force search.
"""

import sys
import os
import json
import numpy as np
import struct

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bindings', 'python'))

from cmfo.core.fractal_algebra_1_1 import FractalUniverse1024
from cmfo.core.hyper_metrics import HyperMetrics
from cmfo.core.positional import PositionalAlgebra

def extract_nonce_from_header(header_bytes):
    """Extract 4-byte nonce from 80-byte header"""
    return struct.unpack("<I", header_bytes[76:80])[0]

def set_nonce_in_header(header_bytes, nonce):
    """Set 4-byte nonce in 80-byte header"""
    h = bytearray(header_bytes)
    h[76:80] = struct.pack("<I", nonce)
    return bytes(h)

def compute_phase_contribution(nibbles, positions):
    """
    Compute phase contribution of specific nibble positions.
    Phase = angle(sum of e^(i*pi/4 * class))
    """
    from cmfo.core.fractal_algebra_1_1 import NibbleAlgebra
    
    vectors = []
    for pos in positions:
        if pos < len(nibbles):
            n = nibbles[pos]
            c, _ = NibbleAlgebra.canon_4(n)
            c_class = NibbleAlgebra.class_projection_8(c)
            vectors.append(np.exp(1j * (np.pi/4) * c_class))
    
    sum_vec = np.sum(vectors)
    angle = np.angle(sum_vec)
    if angle < 0:
        angle += 2*np.pi
    return angle / (2*np.pi)

def geometric_nonce_search(header_template, target_phase=0.949, max_candidates=10000):
    """
    Search for nonce using geometric constraints.
    
    Strategy:
    1. Compute phase of fixed parts
    2. Calculate required nonce phase
    3. Generate candidates matching that phase
    4. Return top candidates
    """
    # Pad header to 1024 bits
    padded = header_template + b'\x00' * (128 - len(header_template))
    u_template = FractalUniverse1024(padded)
    
    # Apply quadratic transform
    delta_quad = (np.arange(256)**2 % 16).astype(int)
    u_trans = PositionalAlgebra.apply(u_template, delta_quad)
    
    # Compute phase of template (with nonce=0)
    v_template = HyperMetrics.compute_7d(u_trans)
    phase_template = v_template[5]
    
    print(f"  Template phase (nonce=0): {phase_template:.4f}")
    print(f"  Target phase: {target_phase:.4f}")
    print(f"  Required delta: {target_phase - phase_template:.4f}")
    
    # Nonce occupies bytes 76-79 (nibbles 152-159 in padded representation)
    # In 1024-bit (256 nibble) space after padding, nonce is at nibbles 152-159
    
    # Generate candidates
    # Simple strategy: try nonces and measure their phase contribution
    candidates = []
    
    for nonce in range(0, min(2**32, max_candidates)):
        # Set nonce
        test_header = set_nonce_in_header(header_template, nonce)
        test_padded = test_header + b'\x00' * (128 - len(test_header))
        u_test = FractalUniverse1024(test_padded)
        u_test_trans = PositionalAlgebra.apply(u_test, delta_quad)
        
        v_test = HyperMetrics.compute_7d(u_test_trans)
        phase_test = v_test[5]
        
        # Distance to target
        dist = abs(phase_test - target_phase)
        
        candidates.append((nonce, phase_test, dist))
        
        if nonce % 1000 == 0 and nonce > 0:
            print(f"  Searched {nonce} candidates...")
    
    # Sort by distance to target
    candidates.sort(key=lambda x: x[2])
    
    return candidates[:10]  # Top 10

def test_geometric_solver():
    print("="*60)
    print("   GEOMETRIC NONCE SOLVER - PROOF OF CONCEPT")
    print("="*60)
    
    # Load a known golden block
    data_path = os.path.join(os.path.dirname(__file__), 'mining_dataset.json')
    with open(data_path) as f:
        dataset = json.load(f)
    
    # Take first hard sample
    sample = dataset['diff_16'][0]
    golden_header = bytes.fromhex(sample['header_hex'])
    golden_nonce = extract_nonce_from_header(golden_header)
    
    print(f"\n[Test Case]")
    print(f"Golden nonce: {golden_nonce} (0x{golden_nonce:08x})")
    print(f"Zeros: {sample['zeros']}")
    
    # Create template (nonce = 0)
    template = set_nonce_in_header(golden_header, 0)
    
    # Compute actual phase of golden solution
    padded_golden = golden_header + b'\x00' * (128 - len(golden_header))
    u_golden = FractalUniverse1024(padded_golden)
    delta_quad = (np.arange(256)**2 % 16).astype(int)
    u_golden_trans = PositionalAlgebra.apply(u_golden, delta_quad)
    v_golden = HyperMetrics.compute_7d(u_golden_trans)
    target_phase = v_golden[5]
    
    print(f"\n[Geometric Search]")
    print(f"Target phase (from golden): {target_phase:.4f}")
    
    # Search
    candidates = geometric_nonce_search(template, target_phase, max_candidates=10000)
    
    print(f"\n[Top 10 Candidates by Phase Proximity]")
    print(f"{'Rank':<6} {'Nonce':<12} {'Phase':<10} {'Distance':<10} {'Match?'}")
    print("-" * 60)
    
    for i, (nonce, phase, dist) in enumerate(candidates, 1):
        match = "✓ GOLDEN" if nonce == golden_nonce else ""
        print(f"{i:<6} {nonce:<12} {phase:.4f}     {dist:.6f}   {match}")
    
    # Check if golden is in top 10
    golden_rank = None
    for i, (nonce, _, _) in enumerate(candidates, 1):
        if nonce == golden_nonce:
            golden_rank = i
            break
    
    print(f"\n[Result]")
    if golden_rank:
        print(f"✓ Golden nonce found at rank {golden_rank}/10")
        print(f"  Search reduction: {10/10000*100:.2f}% of full space")
    else:
        print(f"✗ Golden nonce not in top 10 of {10000} searched")
        print(f"  (May need larger search or refined phase calculation)")

if __name__ == "__main__":
    test_geometric_solver()
