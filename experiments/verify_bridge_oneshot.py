import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from cmfo.storage.bridge import KnowledgeBridge

def verify():
    print("Testing Bridge...")
    bridge = KnowledgeBridge()
    stats = bridge.get_stats()
    print(f"Index Size: {stats['vectors_indexed']}")
    
    term = "galopé"
    print(f"Querying '{term}'...")
    vec = bridge.get_vector(term)
    
    if vec:
        print("SUCCESS: Vector Retrieved.")
        print(f"Vector: {vec[:3]}...")
    else:
        print("FAILURE: Vector not found.")

if __name__ == "__main__":
    verify()
