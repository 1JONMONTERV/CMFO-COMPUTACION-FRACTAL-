"""
CMFO D19: Metacognitive Audit Demo
==================================
Runs the full Reasoner -> Auditor pipeline.
Demonstrates internal quality control of scientific derivations.
"""

import sys
import os
import random

sys.path.insert(0, os.path.abspath('.'))

from cmfo.cognition.graph import ConceptGraph
from cmfo.cognition.reasoner import Reasoner
from cmfo.cognition.auditor import MetacognitiveAuditor

DB_PATH = "D:/CMFO_DATA/concepts/computacion.db"

def run_audit():
    # Enforce UTF-8
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
        
    print("Initializing Metacognitive Stack (D19)...")
    
    # 1. Load Brain
    graph = ConceptGraph("computacion", DB_PATH)
    graph.load()
    
    reasoner = Reasoner(graph)
    auditor = MetacognitiveAuditor(vocabulary_manager=None) # No vocab mgr for this demo, assumes implicit
    
    # 2. Find a Trace
    print("\n[Thinking] Finding a derivation trace...")
    trace = None
    start_node = None
    end_node = None
    
    # Search for a non-trivial path
    nodes = list(graph.nodes.keys())
    for _ in range(50):
        s = random.choice(nodes)
        
        # Try to find a neighbor of a neighbor (Path Len 2) to make it interesting
        node_s = graph.get_node(s)
        if not node_s or not node_s.edges_out: continue
        
        # Hop 1
        hop1 = node_s.edges_out[0][1]
        node_h1 = graph.get_node(hop1)
        if not node_h1 or not node_h1.edges_out: continue
        
        # Hop 2
        hop2 = node_h1.edges_out[0][1]
        
        if s != hop2:
            trace = reasoner.explain_connection(s, hop2)
            if trace:
                start_node = s
                end_node = hop2
                break
                
    if not trace:
        print("Could not find a suitable trace in reasonable time.")
        return

    print(f"Path Found: '{start_node}' -> '{end_node}' ({len(trace.steps)} steps)")

    # 3. Audit
    print("\n[Auditing] Verifying logic integrity...")
    result = auditor.audit(trace)
    
    # 4. Report
    print("\n--- 🛡️ VERIFICATION REPORT ---")
    print(f"Global Confidence (C): {result.confidence:.2f}")
    print(f"Solidity (S):          {result.solidity_score:.2f}")
    print(f"Domain Integrity (D):  {result.domain_integrity_score:.2f}")
    print(f"Evidence Strength (E): {result.evidence_score:.2f}")
    
    print("\n[Flags]")
    for flag, status in result.flags.items():
        icon = "🔴" if status else "🟢"
        print(f"  {icon} {flag}: {status}")
        
    if result.issues:
        print("\n[Issues]")
        for issue in result.issues:
            print(f"  ⚠️ {issue}")
            
    print("\n[Interpretation]")
    if result.confidence > 0.8:
        print("✅ scientifically Valid Derivation")
    elif result.confidence > 0.5:
        print("⚠️ Plausible Association (Hypothetical)")
    else:
        print("❌ Speculative / Weak Connection")

    # 5. D18 Integration (The Voice)
    from cmfo.decision.renderer import StructuralRenderer
    print("\n--- [ D18 RENDERER OUTPUT ] ---")
    renderer = StructuralRenderer(graph)
    explanation = renderer.explain_proof(trace, audit_result=result)
    print(explanation)

if __name__ == "__main__":
    run_audit()
