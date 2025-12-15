
import random
import sys
from engine import AutopoieticUniverse

def run_convergence():
    print("\n--- EXPERIMENT 1: CONVERGENCE (The Attractor) ---")
    print("Does the universe inevitably evolve to the same Alpha?")
    
    TRIALS = 20 # Reduced for speed in terminal demo
    results = []
    
    for i in range(TRIALS):
        # Random initialization of "Phi" (Simulating different geometry seeds)
        # Real Phi is 1.618. We try 1.0 to 2.0
        seed_phi = random.uniform(1.0, 3.0) 
        
        sim = AutopoieticUniverse(seed_phi=seed_phi)
        hist = sim.run(steps=30)
        
        final_alpha = hist[-1]['alpha']
        results.append((seed_phi, final_alpha))
        print(f"  Trial {i+1}: Seed Phi={seed_phi:.4f} -> Final Alpha={final_alpha:.6e}")
        
    # Analyze
    alphas = [r[1] for r in results]
    avg_alpha = sum(alphas) / len(alphas)
    variance = sum((x - avg_alpha)**2 for x in alphas) / len(alphas)
    
    print("-" * 40)
    print(f"Mean Final Alpha: {avg_alpha:.6e}")
    print(f"Variance: {variance:.6e}")
    
    if variance < 1e-8:
        print("RESULT: STRONG CONVERGENCE DETECTED. Universal Attractor Confirmed.")
    else:
        print("RESULT: Weak or No Convergence. Laws depend on initial seed.")

if __name__ == "__main__":
    run_convergence()
