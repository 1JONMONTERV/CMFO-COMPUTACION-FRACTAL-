
import sys
sys.path.insert(0, '../bindings/python')
from cmfo.genesis import derive_phi
from engine import AutopoieticUniverse

def run_correspondence():
    print("\n--- EXPERIMENT 3: CORRESPONDENCE (Reality Check) ---")
    print("Is our Universe (Standard Model) a Fixed Point?")
    
    # 1. Initialize with EXACT calculated values from Genesis Phase 9
    real_phi = derive_phi()
    # Approx Alpha from Genesis demo (Target)
    target_alpha = 1.0 / 137.035999 
    
    print(f"  Target Alpha (Reality): {target_alpha:.8e}")
    
    # Run simulation starting AT target
    sim = AutopoieticUniverse(seed_phi=real_phi, seed_alpha=target_alpha)
    hist = sim.run(steps=50)
    
    final_alpha = hist[-1]['alpha']
    delta = abs(final_alpha - target_alpha)
    percent_error = (delta / target_alpha) * 100
    
    print(f"  Final Alpha (Simulated): {final_alpha:.8e}")
    print(f"  Drift: {delta:.2e} ({percent_error:.4f}%)")
    
    if percent_error < 5.0:
        print("RESULT: MATCH. Our universe is a stable solution to the fractal equation.")
    else:
        print("RESULT: DRIFT. Our universe might be a transient state or equation needs tuning.")

if __name__ == "__main__":
    run_correspondence()
