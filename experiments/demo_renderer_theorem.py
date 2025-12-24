"""
CMFO D18: Scientific Explainer Demo
===================================
Demonstration of the Pedagogical Renderer.
The system picks a random Theorem from the database and explains it.
"""

import sys
import os
import random

sys.path.insert(0, os.path.abspath('.'))

from cmfo.cognition.graph import ConceptGraph
from cmfo.decision.renderer import StructuralRenderer

DB_PATH = "D:/CMFO_DATA/concepts/computacion.db"

def run_demo():
    # Force UTF-8 for Windows Console
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    print("Initializing Scientific Renderer...")
    
    # 1. Load Knowledge
    graph = ConceptGraph("computacion", DB_PATH)
    graph.load()
    
    # 2. Init Renderer
    renderer = StructuralRenderer(graph)
    
    # 3. Find a Theorem
    print("Scanning for Theorems...")
    theorems = [node for node in graph.nodes.values() if getattr(node, 'c_type', 'concept') == 'theorem']
    
    if not theorems:
        print("No theorems found in loaded graph. Is the graph populated?")
        # Fallback to concepts
        print("Falling back to standard concepts.")
        targets = list(graph.nodes.values())
    else:
        print(f"Found {len(theorems)} theorems.")
        targets = theorems
        
    # 4. Explain random target
    if targets:
        target = random.choice(targets)
        print("\n--- [ SYSTEM EXPLANATION ] ---")
        explanation = renderer.explain_concept(target.term, depth=0.6)
        print(explanation)
        print("------------------------------")
        
        # 5. Explaining a path (if connected)
        print("\n[Derivation Check]")
        # Try to find a neighbor to explain connection
        if target.edges_out:
            neighbor_term = target.edges_out[0][1]
            print(f"Explaining link to: {neighbor_term}")
            # Here we would call reasoner, but for D18 unit test we just show we can render the text
            print(f"Renderer says: '{target.term}' relates to '{neighbor_term}' because [Graph Edge Found].")

if __name__ == "__main__":
    run_demo()
