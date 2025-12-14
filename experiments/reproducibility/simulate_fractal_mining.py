import numpy as np
import time
import math
from invert_mini_sha import mini_sha_round, inverse_mini_sha, FractalState, PHI, DIM

# Use the existing robust logic from invert_mini_sha
# but wrap it in a "Mining Simulation" context

def brute_force_solver(target_hash, salt, max_attempts=100000):
    """
    Standard Mining: Guess random inputs until Hash(Input) is close to Target.
    (Simplified for float vectors: distance < epsilon)
    """
    start_time = time.time()
    
    # We relax the condition for brute force because exact float matching is impossible randomly
    # We look for "similarity" > 99%
    target_norm = target_hash.vec / np.linalg.norm(target_hash.vec)
    
    for i in range(max_attempts):
        guess = FractalState() # Random init
        result = mini_sha_round(guess, salt)
        
        # Check alignment (Cosine similarity)
        res_norm = result.vec / np.linalg.norm(result.vec)
        similarity = np.dot(target_norm, res_norm)
        
        if similarity > 0.9999: # Approximate collision
            return i+1, guess, time.time() - start_time
            
    return max_attempts, None, time.time() - start_time

def geometric_solver(target_hash, salt):
    """
    CMFO Mining: Apply the Inverse Operator Gamma^-1.
    No guessing. Direct path.
    """
    start_time = time.time()
    
    # The "Mining" is just the Inverse Function
    # In a full SHA, this would be layered. Here it is 1 layer.
    recovered_state = inverse_mini_sha(target_hash, salt)
    
    return 1, recovered_state, time.time() - start_time

def run_simulation():
    print("=== CMFO vs Standard Mining Simulation ===")
    print("Scenario: Find Nonce 'X' such that Hash(X, Salt) = Target\n")
    
    # 1. Setup the "Block"
    true_secret = FractalState([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5]) # The "Golden Nonce"
    salt = FractalState([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])          # Block Header
    
    target_hash = mini_sha_round(true_secret, salt)
    
    print(f"Target Hash: {target_hash.vec[:3]}...")
    
    # 2. Run Brute Force (Standard Bitcoin Mining)
    print("\n--- Method A: Brute Force (Standard) ---")
    print("Attempting to guess the nonce...")
    bf_steps, bf_result, bf_time = brute_force_solver(target_hash, salt, max_attempts=50000)
    
    if bf_result:
        print(f"SUCCESS: Found match after {bf_steps} attempts.")
        print(f"Time: {bf_time:.4f}s")
    else:
        print(f"FAILED: Gave up after {bf_steps} attempts.")
        print(f"Time: {bf_time:.4f}s")
        
    # 3. Run Geometric Inversion (CMFO Mining)
    print("\n--- Method B: Geometric Inversion (CMFO) ---")
    print("Applying Inverse Operator...")
    geo_steps, geo_result, geo_time = geometric_solver(target_hash, salt)
    
    # Validate result
    check_hash = mini_sha_round(geo_result, salt)
    error = np.linalg.norm(check_hash.vec - target_hash.vec)
    
    if error < 1e-15:
        print(f"SUCCESS: Analytical solution found in {geo_steps} step.")
        print(f"Time: {geo_time:.6f}s")
        print(f"Precision Error: {error:.2e}")
        
        # Speedup Calculation
        ratio = bf_time / geo_time if bf_time > 0 else 1.0
        if not bf_result:
            print(f"\nSpeedup: INFINITE (Brute force failed, CMFO succeeded)")
        else:
            print(f"\nSpeedup: {ratio:.1f}x faster")
            
    else:
        print(f"FAILED: Inversion mismatch (Error: {error:.2e})")

    print("\nCONCLUSION:")
    print("Standard Mining is probabilistic (Gambling).")
    print("CMFO Mining is deterministic (Geometry).")

if __name__ == "__main__":
    run_simulation()
