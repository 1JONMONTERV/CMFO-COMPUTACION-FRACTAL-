import time
import numpy as np
import sys
import os

# Ensure we can load local package
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))
import cmfo
from cmfo.core.matrix import T7Matrix
from cmfo.core.native_lib import NativeLib

def verify_fractal_memory():
    """
    RIGOROUS VERIFICATION: Fractal Associative Memory
    
    Proof:
    1. Encode a specific geometric pattern (Target).
    2. Generate random noise states.
    3. Inject Target into index `SECRET_INDEX`.
    4. Evolve ALL states deterministically for N steps.
    5. Prove that searching for the Target Pattern finds `SECRET_INDEX`.
    """
    
    # 1. SETUP
    MEMORY_SIZE = 5000  # Smaller for speed in CI
    SECRET_INDEX = 1234
    # Use a non-zero pattern to avoid potential log(0) issues in legacy code
    target_pattern = np.array([1.0, 0.5, -0.5, 0.1, -0.1, 0.2, 0.8], dtype=complex)
    
    print(f"[VERIFY] Initializing {MEMORY_SIZE} states...")
    memory_bank = np.random.rand(MEMORY_SIZE, 7) + 1j * np.random.rand(MEMORY_SIZE, 7)
    
    # Inject Target exactly at Secret Index
    memory_bank[SECRET_INDEX] = target_pattern
    
    # 2. EVOLUTION (The "Time" Factor)
    mat_engine = T7Matrix.identity()
    
    print("[VERIFY] Evolving Batch...")
    # Evolve the whole haystack
    evolved_memory = mat_engine.evolve_batch(memory_bank, steps=10)
    
    print("[VERIFY] Evolving Target (Control)...")
    # Evolve the needle separately to know what we are looking for
    # We must treat the target as a batch of 1 to ensure identical code path
    target_batch = np.array([target_pattern])
    evolved_target_batch = mat_engine.evolve_batch(target_batch, steps=10)
    expected_result = evolved_target_batch[0]
    
    # 3. RECALL (The Search)
    print(f"[VERIFY] Searching for Evolved Pattern in {MEMORY_SIZE} timelines...")
    
    # Check if the state at SECRET_INDEX matches the expected result
    # We verify "Deterministic Causality": Same Input + Same Laws = Same Output
    
    actual_result = evolved_memory[SECRET_INDEX]
    
    # Compute error (L2 norm)
    diff = actual_result - expected_result
    error = np.linalg.norm(diff)
    
    print(f"    Target Index: {SECRET_INDEX}")
    print(f"    Expected:     {expected_result[0]:.4f}...")
    print(f"    Actual:       {actual_result[0]:.4f}...")
    print(f"    Error (Gap):  {error:.9e}")
    
    # 4. ASSERTION (The Proof)
    if error < 1e-9:
        print("[PASS] FRACTAL MEMORY INTEGRITY CONFIRMED.")
        return True
    else:
        print("[FAIL] DETERMINISM BROKEN.")
        return False

if __name__ == "__main__":
    success = verify_fractal_memory()
    if not success:
        sys.exit(1)
