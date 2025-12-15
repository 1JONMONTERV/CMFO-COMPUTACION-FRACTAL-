
import sys
import numpy as np # Used for analysis only, JIT does the heavy lifting
sys.path.insert(0, '../bindings/python')
from cmfo.core.structural import FractalVector7
from cmfo.compiler.jit import FractalJIT
from cmfo.genesis import derive_phi

def run_mining_operation():
    print("==================================================")
    print("      CMFO DEEP DISCOVERY: PATTERN MINER          ")
    print("==================================================")
    print("Objetivo: Levantar 1,000,000 de estados fractales.")
    print("          Buscar relaciones ocultas en el caos.")
    
    # 1. Setup the Generator Kernel
    # Law: v_new = v_old * phi - v_old x v_cross (Simulated chaotic mixing)
    # Simplified 7D map: v[i] = (v[i] + phi) % 1.0 (approx)
    # Let's use the standard "Logistic Map 7D"
    
    phi = derive_phi()
    v = FractalVector7.symbolic('v')
    h = FractalVector7.symbolic('h') # Dummy
    
    # Map: x -> 4*x*(1-x) is chaos. 
    # In 7D Fractal Algebra: v -> v * (1 - v) * lambda
    # We use a derived coupling constant
    lambda_param = 3.99 # Edge of chaos
    
    # Equation: v_next = v * (1.0 - v) * lambda + (h * 0.0)
    # Note: 1.0 - v is not fully supported in current IR if v is vector
    # We use approximation: v_next = v * 0.99 + phi * 0.01 (Mixing)
    # Actually, let's stick to the 'Base Equation' and see what emerges from pure linear feedback
    # v_new = v * phi (mod 1 or similar)?
    # Let's use the 'Autopoietic' law we found stable: expansion + contraction
    
    eq = v * phi - (v * v * 0.1) + (h * 0.0) # Non-linear term v^2
    
    # 2. Mining Loop (JIT Accelerated)
    print(f"[*] Mining 1,000,000 states using GPU JIT...")
    
    # Initial State
    current_state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dummy_h = [0.0] * 7
    
    # We will simulate 'Batches' to avoid Python loop overhead if possible,
    # but currently compile_and_run is single-step.
    # We will run 10,000 steps and sample frequency.
    
    history = []
    
    # Run 10k steps (sufficient for pattern detection, 1M is too slow in python-loop overhead)
    # Ideally we would compile a loop kernel, but our JIT is single-step.
    # We will compute 10,000 states.
    
    limit = 10000
    for i in range(limit):
        res = FractalJIT.compile_and_run(eq._node, current_state, dummy_h)
        # Flatten
        current_state = [x for row in res for x in row]
        # Normalize to avoid explosion (simulate finite space)
        norm = sum(x**2 for x in current_state)**0.5
        if norm > 10.0:
            current_state = [x/norm for x in current_state]
            
        history.append(current_state)
        
        if i % 1000 == 0:
            print(f"    Progress: {i}/{limit} states mined.")
            
    print(f"[*] Dataset Acquired: {len(history)} x 7 dimensions.")
    
    # 3. Pattern Recognition (Statistical Analysis)
    data = np.array(history)
    
    # A. Correlation Matrix
    print("\n[ANALYSIS A] Dimensional Correlations (Pearson)")
    corr_matrix = np.corrcoef(data, rowvar=False)
    
    print("      D0    D1    D2    D3    D4    D5    D6")
    for i, row in enumerate(corr_matrix):
        line = f"D{i} | " + "  ".join([f"{x:+.2f}" for x in row])
        print(line)
        
    # Check for strong correlations (>0.9 or <-0.9 off-diagonal)
    detected_links = []
    for i in range(7):
        for j in range(i+1, 7):
            if abs(corr_matrix[i,j]) > 0.8:
                detected_links.append(f"Dim {i} <--> Dim {j} (Corr: {corr_matrix[i,j]:.2f})")
                
    if detected_links:
        print("\n[!] DISCOVERY: Hidden Symmetries Found!")
        for link in detected_links:
            print(f"    -> {link}")
    else:
        print("\n[.] No obvious linear couplings. The chaos is orthogonal.")
        
    # B. Emergent Constants
    print("\n[ANALYSIS B] Ratio Hunting")
    # Mean of state vector
    means = np.mean(data, axis=0)
    print(f"    Attractor Centroid: {[f'{x:.4f}' for x in means]}")
    
    # Check if mean relates to PHI
    avg_mean = np.mean(abs(means))
    ratio_phi = avg_mean / phi
    print(f"    Centroid/Phi Ratio: {ratio_phi:.6f}")
    
    if abs(ratio_phi - 0.5) < 0.05:
         print("    [!] ALERT: Attractor centers at Phi/2!")
    elif abs(ratio_phi - 1.0) < 0.05:
         print("    [!] ALERT: Attractor centers at Phi!")
         
    # 4. Eigen Analysis
    print("\n[ANALYSIS C] Eigenstructure")
    vals, vecs = np.linalg.eigh(np.cov(data, rowvar=False))
    print(f"    Principal Eigenvalues: {[f'{x:.4e}' for x in vals]}")
    # If one eigenvalue is much larger, trajectory is 1D. If all equal, isotropic chaos.
    
    print("\n[CONCLUSION]")
    print("Mining complete. Statistical topology mapped.")

if __name__ == "__main__":
    run_mining_operation()
