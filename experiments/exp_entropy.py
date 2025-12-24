
import math
import sys
from engine import AutopoieticUniverse

def calculate_shannon_entropy(data, bins=100):
    """
    Calculates Shannon Entropy of the data distribution.
    H = -sum(p * log2(p))
    """
    if not data: return 0.0
    
    min_val = min(data)
    max_val = max(data)
    if min_val == max_val: return 0.0
    
    # Histogram
    hist = [0] * bins
    step = (max_val - min_val) / bins
    total_count = len(data)
    
    for x in data:
        idx = int((x - min_val) / step)
        if idx >= bins: idx = bins - 1
        hist[idx] += 1
        
    entropy = 0.0
    for count in hist:
        if count > 0:
            p = count / total_count
            entropy -= p * math.log2(p)
            
    return entropy

def run_entropy_analysis():
    print("\n--- EXPERIMENT 4: NEGENTROPY (Order Creation) ---")
    print("Does the Fractal Universe create ORDER from CHAOS?")
    
    # Start with high noise (Chaos)
    sim = AutopoieticUniverse(noise_level=0.1) 
    # Force initial random state
    import random
    sim.current_field = [random.uniform(-1, 1) for _ in range(700)] # 100 particles x 7 dims
    
    initial_entropy = calculate_shannon_entropy(sim.current_field)
    print(f"  T=0 (Chaos): Entropy = {initial_entropy:.4f} bits")
    
    # Evolve
    hist_entropy = []
    for i in range(1, 21):
        sim.step()
        e = calculate_shannon_entropy(sim.current_field)
        hist_entropy.append(e)
        print(f"  T={i:<2}: Entropy = {e:.4f} bits {'(Decreasing)' if e < initial_entropy else ''}")
        
    final_entropy = hist_entropy[-1]
    delta = initial_entropy - final_entropy
    
    print("-" * 40)
    print(f"Initial Entropy: {initial_entropy:.4f}")
    print(f"Final Entropy:   {final_entropy:.4f}")
    print(f"Negentropy (Order Created): {delta:.4f} bits")
    
    if delta > 0.5:
        print("RESULT: CONFIRMED. System acts as a Maxwell's Demon (Generates Structure).")
    else:
        print("RESULT: Inconclusive. System remains chaotic or stable.")

if __name__ == "__main__":
    run_entropy_analysis()
