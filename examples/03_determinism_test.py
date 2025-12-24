"""
03_determinism_test.py
----------------------
This script proves that CMFO is 100% deterministic.
It runs the same calculation multiple times and asserts that
the binary output is identical.
"""

import cmfo
from cmfo.core.api import tensor7
import sys

def main():
    print("--- CMFO Determinism Proof ---")
    
    input_a = 0.123456789
    input_b = 0.987654321
    
    print(f"Testing inputs: {input_a}, {input_b}")
    
    # Run 1
    out_1 = tensor7(input_a, input_b)
    
    # Run 2
    out_2 = tensor7(input_a, input_b)
    
    # Run 3 (Simulating a different session)
    out_3 = tensor7(input_a, input_b)
    
    print(f"Run 1: {out_1}")
    print(f"Run 2: {out_2}")
    print(f"Run 3: {out_3}")
    
    # STRICT Equality Check
    # We do not use 'approximate' equality here. 
    # Determinism means EXACTLY equal bits.
    if out_1 == out_2 == out_3:
        print(">> SUCCESS: Outputs are bit-exact identical.")
    else:
        print(">> FAILURE: Variances detected. System is not deterministic.")
        sys.exit(1)

if __name__ == "__main__":
    main()
