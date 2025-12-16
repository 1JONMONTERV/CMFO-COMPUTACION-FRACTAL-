"""
CMFO D16: Reasoning Demo
========================
Demonstrates the Architect in action.
1. Hydrates Graph from 'Computacion.db'.
2. Infers relationships.
3. Finds a proof/path between two concepts.
"""

import sys
import os
import random

sys.path.insert(0, os.path.abspath('.'))

from cmfo.cognition.graph import ConceptGraph
from cmfo.cognition.reasoner import Reasoner

DB_PATH = "D:/CMFO_DATA/concepts/computacion.db"

def run_demo():
    print("Initializing Architect (Reasoning Engine)...")
    
    if not os.path.exists(DB_PATH):
        print(f"Error: DB not found at {DB_PATH}")
        return

    # 1. Build Graph
    graph = ConceptGraph("computacion", DB_PATH)
    graph.load()
    
    if len(graph.nodes) < 2:
        print("Not enough nodes for reasoning.")
        return

    # 2. Init Reasoner
    architect = Reasoner(graph)
    
    # 3. Deterministic Query
    print("\nSearching for a reasoning chain...")
    
    found = False
    # Iterate through all nodes to find one with a path
    for start_node_id, start_node in graph.nodes.items():
        if not start_node.edges_out: continue
        
        # Pick a target from neighbors (BFS is fast enough to check reachability)
        # We just pick a direct neighbor for V1 demo to be safe, 
        # or a neighbor of a neighbor.
        
        # Le's try to find a path of length 2 if possible for coolness
        for rel1, mid in start_node.edges_out:
            proof = architect.explain_connection(start_node_id, mid)
            if proof:
                print(f"\n[QUERY]: Why is '{start_node_id}' connected to '{mid}'?")
                print("[PROOF]:")
                for step in proof:
                    print(f"  {step}")
                found = True
                break
        if found: break
            
    if not found:
        print("No connected pairs found (Graph appears disjoint).")

if __name__ == "__main__":
    run_demo()
