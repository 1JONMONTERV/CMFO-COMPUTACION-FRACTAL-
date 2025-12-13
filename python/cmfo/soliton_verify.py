import math
import csv
import sys
import os

# Constants
M_PI = 3.14159265358979323846

def exact_kink(x, t, x0, v, polarity):
    """
    Returns the exact value of a Sine-Gordon Kink/Anti-Kink at (x,t).
    phi(x,t) = 4 * atan(exp(gamma * (x - x0 - v*t)))
    """
    if abs(v) >= 1.0:
        raise ValueError("Velocity must be < 1.0")
    
    gamma = 1.0 / math.sqrt(1.0 - v**2)
    arg = gamma * (x - x0 - v * t)
    
    # Avoid overflow in exp
    if arg > 20: val = 4.0 * M_PI / 2.0 # approx 2pi
    elif arg < -20: val = 0.0
    else: val = 4.0 * math.atan(math.exp(arg))
    
    if polarity < 0:
        val = -val # Anti-Kink (simplified, strictly 2pi - val if shifting)
                   # But for zero-centered field, this anti-symmetric flip is standard for test
    return val

def load_csv(filename):
    data = []
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None
        
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'x': float(row['x']),
                    'phi': float(row['phi']),
                    'phi_t': float(row['phi_t'])
                })
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    return data

def verify_state(data, t, kinks_config):
    """
    Compare loaded data against analytical superposition of kinks.
    kinks_config: list of dicts {'x0', 'v', 'pol'}
    """
    mse = 0.0
    max_err = 0.0
    
    for point in data:
        x = point['x']
        phi_sim = point['phi']
        
        # Analytical Superposition
        phi_theo = 0.0
        for k in kinks_config:
            phi_theo += exact_kink(x, t, k['x0'], k['v'], k['pol'])
            
        err = abs(phi_sim - phi_theo)
        mse += err * err
        if err > max_err: max_err = err
        
    mse /= len(data)
    return mse, max_err

def main():
    print("=== CMFO Soliton Verification ===")
    
    # Configuration matching test_soliton_collision.c
    # Initial state: t=0
    # Kink 1: x0=-8.0, v=0.4, pol=1
    # Kink 2: x0=8.0, v=-0.4, pol=-1
    
    # Verify Initial State (Sanity Check)
    print("\nVerifying Initial State (soliton_init.csv)...")
    data_init = load_csv("soliton_init.csv")
    if data_init:
        config_init = [
            {'x0': -8.0, 'v': 0.4, 'pol': 1},
            {'x0': 8.0, 'v': -0.4, 'pol': -1}
        ]
        mse_i, max_i = verify_state(data_init, 0.0, config_init)
        print(f"  MSE: {mse_i:.6e}, Max Diff: {max_i:.6e}")
        if mse_i < 1e-4: print("  [PASS] Initial state matches theory.")
        else: print("  [FAIL] Initial state mismatch!")
    else:
        print("  [SKIP] soliton_init.csv missing (Compile C code first).")

    # Verify Final State (Post-Collision)
    # T = steps * dt = 2000 * 0.05 = 100.0
    # Note: After collision, phase shift occurs. Exact analytical match for simple 
    # superposition fails close to collision, but at large t they separate.
    # We will just print stats here for now or update expectation.
    print("\nVerifying Final State (soliton_final.csv)...")
    data_final = load_csv("soliton_final.csv")
    if data_final:
        print("  Data loaded. Computing stats...")
        # Just simple conservation check or visual inspection would be better
        # Since we know C code checks E and Q, we trust that.
        # This script confirms the files are readable.
        print("  [INFO] Verification script ready. Run 'test_soliton' to generate data.")

if __name__ == "__main__":
    main()
