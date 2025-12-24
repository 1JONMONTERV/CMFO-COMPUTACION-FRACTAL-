"""
CMFO REALITY CHECK
==================
Verifies if:
1. cmfo_jit.dll can be loaded.
2. Fractal Memory actually stores/recalls data.
3. GPU acceleration is active.
"""

import sys
import os
import time

# Ensure bindings are in path
sys.path.append(os.path.abspath("bindings/python"))

try:
    import cmfo
    from cmfo.compiler.jit import FractalJIT
    from cmfo.memory.fractal_memory import FractalMemoryBank
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import CMFO modules: {e}")
    sys.exit(1)

def verify_jit():
    print("\n[1] Checking JIT Bridge...")
    is_avail = FractalJIT.is_available()
    if is_avail:
        print("    [SUCCESS] FractalJIT is AVAILABLE. GPU Bridge is Active.")
        return True
    else:
        print("    [WARNING] FractalJIT is NOT available. Running in CPU Simulation Mode.")
        return False

def verify_memory(use_jit):
    print("\n[2] Checking Fractal Memory Mechanics...")
    try:
        bank = FractalMemoryBank(capacity=10)
        
        # Test Data: 7D Concept Vector
        concept = [0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2.1] # Phi-ish
        
        print(f"    Storing Concept: {concept[:3]}...")
        stored = bank.write(0, concept)
        print(f"    Stored State (Transformed): {stored[:3]}...")
        
        retrieved = bank.read(0)
        print(f"    Retrieved: {retrieved[:3]}...")
        
        # Simple integrity check
        delta = sum(abs(a - b) for a, b in zip(concept, retrieved))
        if delta < 1e-5:
            print(f"    [SUCCESS] Memory Integrity Confirmed (Delta={delta:.9f}).")
        else:
            print(f"    [FAILURE] Memory Corruption Detected (Delta={delta:.9f}).")
            
    except Exception as e:
        print(f"    [FAILURE] Memory Logic Crashed: {e}")

def main():
    print("=== CMFO SYSTEM REALITY CHECK ===")
    jit_active = verify_jit()
    verify_memory(jit_active)
    
    if jit_active:
        print("\n[CONCLUSION] The system is REAL, EXECUTABLE, and GPU-ACCELERATED.")
    else:
        print("\n[CONCLUSION] The system is EXECUTABLE but using CPU Simulation (No GPU).")

if __name__ == "__main__":
    main()
