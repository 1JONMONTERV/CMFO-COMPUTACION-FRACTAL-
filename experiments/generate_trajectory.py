import json
import numpy as np
import os
import sys

# Load CMFO Native
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))
import cmfo
from cmfo.core.matrix import T7Matrix

def generate_web_data():
    print("Generating Simulation Data for Web Visualization...")
    
    # 1. Setup Simulation
    steps = 50
    mat = T7Matrix() # Identity (Standard Phi Evolution)
    
    # Initial State: "The Seed"
    # A simple clean seed to show oscillation
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    trajectory = []
    
    # 2. Run Loop using C++ Evolve
    # We want INTERMEDIATE states for plotting, so we run 1 step at a time
    # (Speedup is less relevant here given I/O, but we use the engine for correctness)
    
    current_state = state
    for i in range(steps):
        # Record
        entry = {"step": i}
        for dim in range(7):
            entry[f"d{dim}"] = float(current_state[dim].real) # Plot Real part
            
        trajectory.append(entry)
        
        # Evolve
        # Note: Evolve returns NEW state
        current_state = mat.evolve_state(current_state, steps=1)
        
    # 3. Save JSON
    output_dir = os.path.join(os.path.dirname(__file__), "..", "web", "static", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "simulation_data.json")
    
    with open(output_path, "w") as f:
        json.dump(trajectory, f, indent=2)
        
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    generate_web_data()
