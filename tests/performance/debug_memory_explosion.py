
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("bindings/python"))

from cmfo.core.matrix import T7Matrix

def hunt_divergence():
    print("=== Hunting Divergence ===")
    
    # Check 1000 random vectors
    for i in range(1000):
        # Generate with slightly wider range to provoke instability
        # random.randn gives normal distribution (unbounded)
        state = np.random.randn(7) + 1j * np.random.randn(7)
        
        # Check initial imag part magnitude
        if np.max(np.abs(state.imag)) > 2.0:
            print(f"Found high-energy state #{i}: ImagMax={np.max(np.abs(state.imag)):.2f}")
            
            # Trace this one
            v = state
            mat = T7Matrix.identity()
            
            exploded = False
            for step in range(50):
                try:
                    v_next = mat.evolve_state(v, steps=1)
                    mag = np.linalg.norm(v_next)
                    
                    if not np.isfinite(mag) or mag > 1e10:
                        print(f"  !!! EXPLODED at step {step} !!!")
                        exploded = True
                        break
                    v = v_next
                except Exception as e:
                    print(f"  !!! EXCEPTION at step {step}: {e}")
                    exploded = True
                    break
            
            if exploded:
                return
    
    print("No explosion found in 1000 samples. The instability might be rare or require specific conditions.")

if __name__ == "__main__":
    hunt_divergence()
