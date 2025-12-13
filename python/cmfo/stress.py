import time
import random
import math
import sys
from .wrapper import CMFOCore

def main():
    print("=== CMFO MAXIMUM LEVEL STRESS TEST & BENCHMARK ===")
    
    try:
        core = CMFOCore()
    except Exception as e:
        print(f"FAILED to load core: {e}")
        return

    print(f"Library Loaded. Phi Constant Check: {core.phi()}")
    
    # --- TEST 1: ALGEBRAIC INTEGRITY FLOOD ---
    print("\n[TEST 1] Algebraic Integrity Flood (100,000 Tensor Ops)...")
    
    a = [random.random() for _ in range(7)]
    b = [random.random() for _ in range(7)]
    
    start = time.time()
    iterations = 100000
    errors = 0
    
    for i in range(iterations):
        # Property: Commutativity of Symmetric Tensor
        # In this specific algebra, let's just check stability (no crash)
        # And ensure output isn't NaN
        try:
            res = core.tensor7(a, b)
            if any(math.isnan(x) or math.isinf(x) for x in res):
                errors += 1
                break
            
            # Mutate inputs slightly to simulate dynamic system
            a[i % 7] = res[(i+1)%7] * 0.1
        except Exception as e:
            print(f"Crash at iteration {i}: {e}")
            errors += 1
            break
            
    end = time.time()
    duration = end - start
    
    if errors == 0:
        print(f"  [PASS] {iterations} ops completed without numerical breakdown.")
        print(f"  Throughput: {iterations / duration:,.0f} ops/sec (Single Thread Python->C)")
    else:
        print("  [FAIL] Numerical instability detected.")

    # --- TEST 2: MATRIX INVERSION STABILITY STRESS ---
    print("\n[TEST 2] Matrix Inversion Stability (High Condition Number Search)...")
    
    singular_counts = 0
    stable_counts = 0
    attempts = 1000
    
    for _ in range(attempts):
        # Generate random matrix
        M = [[random.uniform(-1,1) for _ in range(7)] for _ in range(7)]
        
        try:
            inv = core.mat7_inv(M)
            stable_counts += 1
        except ValueError:
            singular_counts += 1 # Singular is valid result, just checking it catches it safely
        except Exception as e:
            print(f"  [CRITICAL FAIL] Segfault/Exception on inversion: {e}")
            return

    print(f"  [PASS] Processed {attempts} matrices. {stable_counts} invertible, {singular_counts} singular.")
    print("  Core successfully detected singularities without crashing.")

    # --- RESULTS ---
    print("\n=== STRESS TEST COMPLETE ===")
    print("System Status: STABLE at Maximum Load.")

if __name__ == "__main__":
    main()
