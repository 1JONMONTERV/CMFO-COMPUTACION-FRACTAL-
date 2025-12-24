
from engine import AutopoieticUniverse

def run_stability():
    print("\n--- EXPERIMENT 2: STABILITY (Chaos Resistance) ---")
    print("Can properties survive quantum noise injection?")
    
    noise_levels = [0.0, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
    
    for noise in noise_levels:
        sim = AutopoieticUniverse(noise_level=noise)
        hist = sim.run(steps=30)
        
        # Check stability of last 10 steps
        last_10 = [h['alpha'] for h in hist[-10:]]
        avg = sum(last_10) / 10
        fluctuation = max(last_10) - min(last_10)
        
        status = "STABLE" if fluctuation < (avg * 0.1) else "UNSTABLE"
        print(f"  Noise {noise:.1e}: Final Alpha={avg:.6e} | Fluctuation={fluctuation:.2e} -> {status}")

if __name__ == "__main__":
    run_stability()
