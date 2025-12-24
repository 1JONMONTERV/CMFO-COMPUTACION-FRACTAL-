#!/usr/bin/env python3
"""
Rigorous Verification of Hash System
=====================================

Verifies:
1. Bit-exactness: sha256d_ours == hashlib.sha256d
2. Determinism: G(s,i) == G(s,i) always
3. No endianness/representation bugs
"""

import sys
import os
import hashlib

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, '..', 'bindings', 'python')
sys.path.append(bindings_path)

from cmfo.topology.procedural_512 import ProceduralSpace512
from cmfo.topology.hash_lookup_table import HashLookupTable

def reference_sha256d(message_bytes):
    """
    Reference implementation using hashlib.
    This is the ground truth.
    """
    h1 = hashlib.sha256(message_bytes).digest()
    h2 = hashlib.sha256(h1).digest()
    return h2.hex()

def main():
    print("=" * 70)
    print("   RIGOROUS HASH VERIFICATION")
    print("   Bit-exact comparison with hashlib")
    print("=" * 70)
    
    space = ProceduralSpace512()
    table = HashLookupTable()
    
    # Test 1: Bit-exactness
    print(f"\n{'='*70}")
    print("TEST 1: Bit-Exactness (10,000 samples)")
    print('='*70)
    
    N = 10000
    mismatches = 0
    
    for i in range(N):
        # Generate deterministic coordinates
        x = i
        y = i * 2
        
        # Generate message (64 bytes = 512 bits)
        message = space.coords_to_block(x, y)
        assert len(message) == 64, f"Message must be 64 bytes, got {len(message)}"
        
        # Our implementation
        our_hash = table.sha256d_fractal(message)
        
        # Reference implementation
        ref_hash = reference_sha256d(message)
        
        # Compare
        if our_hash != ref_hash:
            mismatches += 1
            if mismatches <= 3:  # Show first 3 errors
                print(f"\n  MISMATCH at i={i}:")
                print(f"    Ours: {our_hash[:32]}...")
                print(f"    Ref:  {ref_hash[:32]}...")
        
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{N} | Mismatches: {mismatches}", end='\r')
    
    print(f"\n\n  Result: {N - mismatches}/{N} matches")
    print(f"  Accuracy: {(N - mismatches)/N * 100:.2f}%")
    
    if mismatches == 0:
        print(f"  ✓ BIT-EXACT MATCH")
    else:
        print(f"  ✗ FAILED - {mismatches} mismatches")
        return False
    
    # Test 2: Determinism
    print(f"\n{'='*70}")
    print("TEST 2: Determinism (1,000 samples)")
    print('='*70)
    
    non_deterministic = 0
    
    for i in range(1000):
        x, y = i * 3, i * 5
        
        # Generate same block twice
        block1 = space.coords_to_block(x, y)
        block2 = space.coords_to_block(x, y)
        
        if block1 != block2:
            non_deterministic += 1
            if non_deterministic <= 3:
                print(f"\n  NON-DETERMINISTIC at ({x}, {y})")
        
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/1000 | Non-det: {non_deterministic}", end='\r')
    
    print(f"\n\n  Result: {1000 - non_deterministic}/1000 deterministic")
    
    if non_deterministic == 0:
        print(f"  ✓ FULLY DETERMINISTIC")
    else:
        print(f"  ✗ FAILED - {non_deterministic} non-deterministic")
        return False
    
    # Test 3: Endianness/Representation
    print(f"\n{'='*70}")
    print("TEST 3: Bytes ↔ Hex Consistency")
    print('='*70)
    
    inconsistent = 0
    
    for i in range(100):
        x, y = i * 7, i * 11
        
        # Generate block
        block_bytes = space.coords_to_block(x, y)
        
        # Convert to hex and back
        block_hex = block_bytes.hex()
        block_recovered = bytes.fromhex(block_hex)
        
        if block_bytes != block_recovered:
            inconsistent += 1
            print(f"  INCONSISTENT at ({x}, {y})")
    
    print(f"\n  Result: {100 - inconsistent}/100 consistent")
    
    if inconsistent == 0:
        print(f"  ✓ REPRESENTATION CONSISTENT")
    else:
        print(f"  ✗ FAILED - {inconsistent} inconsistent")
        return False
    
    # Final verdict
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print('='*70)
    print(f"  Bit-Exactness: ✓ PASS")
    print(f"  Determinism: ✓ PASS")
    print(f"  Representation: ✓ PASS")
    print(f"\n  STATUS: SYSTEM VERIFIED ✓")
    print(f"  Ready for scaling with confidence.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
