#!/usr/bin/env python3
"""
Build Golden Block Database
============================

Scans the nonce space to build a comprehensive database of valid
mining solutions. Uses Midstate optimization for maximum speed.

This demonstrates the "Mining as Database" concept:
- Process millions of candidates
- Store only valid solutions
- Query database instead of mining
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, '..', 'bindings', 'python')
sys.path.append(bindings_path)

from cmfo.mining.optimized_distiller import OptimizedDistiller

def main():
    print("=" * 60)
    print("   CMFO GOLDEN BLOCK DATABASE BUILDER")
    print("   Strategy: Scan → Filter → Store")
    print("=" * 60)
    
    # Initialize distiller
    db_path = os.path.join(current_dir, 'golden_blocks_db.json')
    distiller = OptimizedDistiller(database_path=db_path)
    
    # Configuration
    DIFFICULTY = 5  # 5 hex zeros = 20 bits = ~1 in 1M chance
    SCAN_SIZE = 2_000_000  # 2 million nonces
    
    print(f"\n[Config]")
    print(f"  Difficulty: {DIFFICULTY} leading zeros (≈ 1/{16**DIFFICULTY:,} probability)")
    print(f"  Scan Size: {SCAN_SIZE:,} nonces")
    print(f"  Expected Solutions: ~{SCAN_SIZE / (16**DIFFICULTY):.1f}")
    
    # Create base header
    print(f"\n[Header] Creating base template...")
    base_header = distiller.create_base_header(
        version=2,
        prev_hash=b'\xaa' * 32,
        merkle_root=b'\xff' * 32
    )
    print(f"  Header: {base_header.hex()[:32]}...")
    
    # Scan multiple ranges to build database
    print(f"\n{'='*60}")
    print("PHASE 1: Initial Scan")
    print('='*60)
    
    found = distiller.scan_nonce_space(
        base_header=base_header,
        start_nonce=0,
        count=SCAN_SIZE,
        difficulty=DIFFICULTY,
        batch_report=100000
    )
    
    # Save database
    distiller.save_database()
    
    # Demonstrate query
    print(f"\n{'='*60}")
    print("PHASE 2: Database Query Test")
    print('='*60)
    
    if distiller.solutions:
        print("\n[Query] Testing instant lookup...")
        solution = distiller.query_solution()
        
        print(f"\n  ✓ Retrieved solution from database:")
        print(f"    Nonce: {solution['nonce']:,}")
        print(f"    Hash: {solution['hash']}")
        print(f"    Difficulty: {solution['difficulty']} zeros")
        
        print(f"\n[Success] Mining transformed to O(1) database lookup!")
    else:
        print("\n[Info] No solutions found. Try:")
        print("  - Lower difficulty (e.g., 4 zeros)")
        print("  - Increase scan size")
    
    # Statistics
    if distiller.solutions:
        total_scanned = SCAN_SIZE
        total_stored = len(distiller.solutions)
        compression = total_scanned / total_stored if total_stored > 0 else 0
        
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print('='*60)
        print(f"  Blocks Scanned: {total_scanned:,}")
        print(f"  Valid Solutions: {total_stored}")
        print(f"  Compression Ratio: {compression:,.0f}x")
        print(f"  Database: {db_path}")
        print(f"\n  Status: OPERATIONAL ✓")

if __name__ == "__main__":
    main()
