"""
02_tensor7_basic.py
-------------------
The 'Hello World' of Fractal Computing.
Demonstrates how to import the library and perform a basic T7 operation.
"""

import cmfo
from cmfo.core.api import tensor7

def main():
    print("--- CMFO Basic Example ---")
    
    # Define two scalar inputs (representing signal A and signal B)
    # in a real use case, these could be token embeddings.
    a = 1.0
    b = 0.5
    
    print(f"Input A: {a}")
    print(f"Input B: {b}")
    
    # Compute the T7 Tensor Projection
    # This combines the signals using the Golden Ratio metric.
    result = tensor7(a, b)
    
    print(f"T7 Result: {result}")
    
    # Verify against manual formula: (a*b + PHI) / (1 + PHI)
    PHI = (1 + 5 ** 0.5) / 2
    expected = (a * b + PHI) / (1 + PHI)
    
    print(f"Expected:  {expected}")
    
    if abs(result - expected) < 1e-9:
        print(">> SUCCESS: Calculation matches theory.")
    else:
        print(">> FAILURE: Mismatch detected.")

if __name__ == "__main__":
    main()
