#!/usr/bin/env python3
"""
Hash Lookup Table Demo
=======================

Demonstrates the procedural hash table system:
1. Generate blocks from 2^512 space
2. Compute SHA256d for each
3. Build queryable lookup table
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, '..', 'bindings', 'python')
sys.path.append(bindings_path)

from cmfo.topology.hash_lookup_table import HashLookupTable

def main():
    print("=" * 70)
    print("   PROCEDURAL HASH LOOKUP TABLE")
    print("   Block → SHA256d Mapping")
    print("=" * 70)
    
    # Initialize
    db_file = os.path.join(current_dir, 'hash_lookup.json')
    table = HashLookupTable(db_path=db_file)
    
    # Demo 1: Generate entries
    print(f"\n{'='*70}")
    print("PHASE 1: Generating Hash Table")
    print('='*70)
    
    table.generate_entries(count=5000, start_x=0, start_y=0)
    
    # Demo 2: Query by coordinates
    print(f"\n{'='*70}")
    print("PHASE 2: Query by Coordinates")
    print('='*70)
    
    test_coords = [(0, 0), (100, 200), (1000, 5000)]
    
    for x, y in test_coords:
        hash_result, cached = table.query_by_coords(x, y)
        status = "CACHED" if cached else "GENERATED"
        print(f"\n  Coords ({x}, {y})")
        print(f"  Hash: {hash_result[:32]}...")
        print(f"  Status: {status}")
    
    # Demo 3: Search for specific pattern
    print(f"\n{'='*70}")
    print("PHASE 3: Mining (Search for Pattern)")
    print('='*70)
    
    # Search for hashes with 3 leading zeros
    matches = table.find_blocks_with_prefix("000", max_search=10000)
    
    if matches:
        print(f"\n  Sample match:")
        block, hash_val = matches[0]
        print(f"  Block: {block.hex()[:32]}...")
        print(f"  Hash:  {hash_val}")
    
    # Save table
    print(f"\n{'='*70}")
    print("PHASE 4: Persisting Table")
    print('='*70)
    
    table.save_table()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"  Total Entries: {len(table.table):,}")
    print(f"  Space Coverage: {len(table.table)} / 2^512")
    print(f"  Query Speed: O(1) for cached, O(hash) for procedural")
    print(f"\n  Status: LOOKUP TABLE OPERATIONAL ✓")

if __name__ == "__main__":
    main()
