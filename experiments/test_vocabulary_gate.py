"""
CMFO D14: Vocabulary Gate Test
==============================
Verifies strict ontological separation.
1. Checks Forbidden terms in Computation.
2. Checks Vector Projection (Mind Axis suppression).
"""

import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from cmfo.ontology.vocab import VocabularyManager

def verify_gates():
    mgr = VocabularyManager()
    vocab = mgr.get_context("computacion")
    
    if not vocab:
        print("FAIL: Domain 'computacion' not found.")
        return

    print(f"Testing Domain: {vocab.domain.upper()}")
    
    # 1. Prohibition Check
    terms = ["algoritmo", "milagro", "fe", "bit"]
    print("\n[Gate Check]")
    for t in terms:
        valid = vocab.is_valid(t)
        status = "ALLOWED" if valid else "FORBIDDEN"
        print(f"  '{t}': {status}")
        
    # 2. Projection Check
    print("\n[Projection Check]")
    # Vector: [Ex, Ord, Chaos, Act, Conn, Mind, Time]
    # Mind is index 5. Computation sets it to 0.0
    input_vec = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
    
    projected = vocab.get_projection(input_vec)
    
    print(f"  Input Vector:   {input_vec}")
    print(f"  Projected (CS): {projected}")
    
    if projected[5] == 0.0 and projected[6] == 1.0:
        print("SUCCESS: Mind axis flattened, Time preserved.")
    else:
        print("FAILURE: Projection logic incorrect.")

if __name__ == "__main__":
    verify_gates()
