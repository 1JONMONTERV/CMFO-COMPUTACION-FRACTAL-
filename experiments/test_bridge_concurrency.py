"""
CMFO D10 VERIFICATION: CONCURRENT READ
======================================
Tests the KnowledgeBridge ability to read D9 Data 
while ingestion is running in parallel.
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.abspath('.'))

from cmfo.storage.bridge import KnowledgeBridge

def test_bridge():
    print("Initializing Industrial Bridge...")
    bridge = KnowledgeBridge()
    
    print("\n[STATS] Checking Storage Size...")
    stats = bridge.get_stats()
    print(f"Status: {stats['status']}")
    print(f"Terms:  {stats['terms_indexed']}")
    print(f"Vectors:{stats['vectors_indexed']}")
    
    if stats["vectors_indexed"] == 0:
        print("Warning: Storage empty or locked? (Wait for ingestion to flush WAL)")
        
    print("\n[READ TEST] Querying 'galopé' (From Mass Ingestion)...")
    vec = bridge.get_vector("galopé")
    
    if vec:
        print(f"SUCCESS. Found 'galopé'.")
        print(f"Vector dim: {len(vec)}")
        print(f"Sample: {vec[:3]}...")
    else:
        print("FAIL. 'perro' not found (or bridge blocked).")
        
    print("\n[LOOP] Monitoring Ingestion (Ctrl+C to stop)...")
    try:
        start_count = stats["vectors_indexed"]
        while True:
            # Pulse check
            stats = bridge.get_stats()
            curr = stats["vectors_indexed"]
            
            # Check random known term if dictionary allows, 
            # or just monitor count growth
            
            delta = curr - start_count
            print(f"\rVectors: {curr} (+{delta}) | Reading live...", end="")
            
            # Artificial read load
            v = bridge.get_vector("perro")
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nTest stopped.")

if __name__ == "__main__":
    test_bridge()
