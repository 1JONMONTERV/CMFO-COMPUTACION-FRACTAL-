#!/usr/bin/env python3
"""
Binary Hash Database Builder
=============================

Builds the rigorous binary hash database:
1. Generates hashes_by_i.bin (128-byte header + N*32 payload)
2. Builds prefix_index.bin and prefix_lists.bin
3. Verifies structural and cryptographic integrity
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, '..', 'bindings', 'python')
sys.path.append(bindings_path)

from cmfo.topology.binary_hash_db import BinaryHashDB

def main():
    print("=" * 70)
    print("   BINARY HASH DATABASE BUILDER")
    print("   Rigorous Layout Implementation")
    print("=" * 70)
    
    # Configuration
    N = 100000  # Start with 100K for testing
    base_path = os.path.join(current_dir, 'hash_db')
    
    print(f"\n[Config]")
    print(f"  Entries: {N:,}")
    print(f"  Base path: {base_path}")
    print(f"  Prefix bytes: 2 (65,536 buckets)")
    
    # Initialize
    db = BinaryHashDB(base_path=base_path)
    
    # Step 1: Generate hashes
    print(f"\n{'='*70}")
    print("STEP 1: Hash Generation")
    print('='*70)
    
    payload_hash = db.generate_hashes(N, seed=42)
    
    # Step 2: Build prefix index
    print(f"\n{'='*70}")
    print("STEP 2: Prefix Index Construction")
    print('='*70)
    
    db.build_prefix_index()
    
    # Step 3: Verification
    print(f"\n{'='*70}")
    print("STEP 3: Verification")
    print('='*70)
    
    structural_ok = db.verify_structural()
    crypto_ok = db.verify_cryptographic()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    
    file1_size = os.path.getsize(db.file_hashes)
    file2_size = os.path.getsize(db.file_prefix_idx)
    file3_size = os.path.getsize(db.file_prefix_lists)
    total_size = file1_size + file2_size + file3_size
    
    print(f"  File 1 (hashes): {file1_size/1024/1024:.2f} MB")
    print(f"  File 2 (index):  {file2_size/1024:.2f} KB")
    print(f"  File 3 (lists):  {file3_size/1024/1024:.2f} MB")
    print(f"  Total:           {total_size/1024/1024:.2f} MB")
    print(f"\n  Structural: {'✓ PASS' if structural_ok else '✗ FAIL'}")
    print(f"  Cryptographic: {'✓ PASS' if crypto_ok else '✗ FAIL'}")
    
    if structural_ok and crypto_ok:
        print(f"\n  STATUS: DATABASE VERIFIED ✓")
        print(f"  Ready for O(1) lookups")
    else:
        print(f"\n  STATUS: VERIFICATION FAILED ✗")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
