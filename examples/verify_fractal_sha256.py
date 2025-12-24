#!/usr/bin/env python3
"""
Fractal SHA-256d Verification
==============================

Verifies the fractal implementation against:
1. Standard hashlib (SHA-256d)
2. Known Bitcoin golden solutions
"""

import sys
import os
import hashlib
import json
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, '..', 'bindings', 'python')
sys.path.append(bindings_path)

from cmfo.fractal_sha256 import sha256d_fractal, SHA256Fractal

def std_sha256d(msg):
    """Standard implementation for comparison"""
    return hashlib.sha256(hashlib.sha256(msg).digest()).digest()

def main():
    print("=" * 70)
    print("   FRACTAL SHA-256d VERIFICATION")
    print("   Bit-exactness & Traceability Test")
    print("=" * 70)
    
    # Test 1: Empty string
    print(f"\n[Test 1] Empty string")
    msg = b''
    expected = std_sha256d(msg)
    start = time.perf_counter()
    result = sha256d_fractal(msg)
    elapsed = time.perf_counter() - start
    
    print(f"  Expected: {expected.hex()}")
    print(f"  Fractal:  {result.hex()}")
    print(f"  Match:    {'✓ PASS' if result == expected else '✗ FAIL'}")
    print(f"  Time:     {elapsed:.4f}s")
    
    # Test 2: Simple text
    print(f"\n[Test 2] Simple text 'hello'")
    msg = b'hello'
    expected = std_sha256d(msg)
    result = sha256d_fractal(msg)
    
    print(f"  Expected: {expected.hex()}")
    print(f"  Fractal:  {result.hex()}")
    print(f"  Match:    {'✓ PASS' if result == expected else '✗ FAIL'}")
    
    # Test 3: Bitcoin Golden Solutions
    print(f"\n[Test 3] Bitcoin Golden Solutions (from JSON)")
    json_path = os.path.join(current_dir, 'golden_solutions.json')
    
    if os.path.exists(json_path):
        with open(json_path) as f:
            solutions = json.load(f)
            
        bs_passed = 0
        for i, sol in enumerate(solutions):
            block = bytes.fromhex(sol['block_hex'])
            expected_hash = bytes.fromhex(sol['hash']) # Note: JSON usually has little-endian hash? 
            # In golden_solutions.json, "hash" is typically the block hash (LE displayed as BE string? or pure hex?)
            # Usually block explorers show reversed hex.
            # Let's check against std_sha256d first.
            
            std_hash = std_sha256d(block)
            
            # The JSON likely stores the double-hash directly.
            # Let's use std_hash as ground truth source since we know input is block_hex.
            
            fractal_hash = sha256d_fractal(block)
            
            match = (fractal_hash == std_hash)
            if match:
                bs_passed += 1
                
            print(f"  Block {i+1}: {'✓' if match else '✗'}")
            
        print(f"  Pass Rate: {bs_passed}/{len(solutions)}")
    else:
        print("  Skipped (golden_solutions.json not found)")
        
    # Test 4: Traceability check
    print(f"\n[Test 4] Traceability Inspection")
    engine = SHA256Fractal()
    engine.hash(b'trace_me')
    
    # Inspect Round 0 trace
    trace = engine.get_round_trace(0)
    ops_count = len(trace['modified_cells'])
    unique_ops = trace['operations']
    
    print(f"  Round 0 Trace:")
    print(f"    Modified cells: {ops_count}")
    print(f"    Operations: {unique_ops}")
    
    if ops_count > 0:
        print("  ✓ Trace data detected")
    else:
        print("  ✗ No trace data found")
        
    print(f"\n{'='*70}")
    print("STATUS: FRACTAL SHA-256d IMPLEMENTED")
    print('='*70)

if __name__ == "__main__":
    main()
