import math

# Constants
PHI = (1 + math.sqrt(5)) / 2
PI = math.pi
ALPHA = 7.29735256e-3    # Fine Structure Constant (~1/137)
ALPHA_INV = 1 / ALPHA
H_BAR = 1.054571817e-34
C = 299792458.0
G = 6.67430e-11
J_TO_MEV = 6.241509e12

# Planck Values
M_PLANCK_KG = math.sqrt((H_BAR * C) / G) # ~2.17e-8 kg
M_PLANCK_MEV = M_PLANCK_KG * (C**2) * J_TO_MEV # ~1.22e19 GeV

# Experimental Masses (MeV)
PARTICLES = {
    "Electron": {"exp": 0.510998, "n": 51},
    "Muon":     {"exp": 105.6583, "n": 45},
    "Proton":   {"exp": 938.272,  "n": 39}
}

def check_scalings():
    print(f"Planck Mass: {M_PLANCK_MEV:.4e} MeV")
    print("Searching for a UNIFIED geometric operator (Omega) to bridge the gap...\n")
    
    # We are looking for M_real = M_fractal * Omega
    # So Omega = M_real / M_fractal
    
    results = {}
    
    for name, data in PARTICLES.items():
        n = data['n']
        m_real = data['exp']
        m_frac = M_PLANCK_MEV * (PHI ** -n)
        
        # The required correction factor
        required_factor = m_real / m_frac
        results[name] = required_factor
        
        print(f"=== {name} (n={n}) ===")
        print(f"  Required Correction (Omega): {required_factor:.4e}")
        
        # Check against Alpha powers (The most likely candidate for gauge coupling)
        # alpha_power = log(Omega) / log(alpha)
        alpha_p = math.log(required_factor) / math.log(ALPHA)
        print(f"  Matches Alpha^{alpha_p:.3f}")
        
    print("\n=== Unified Analysis ===")
    print("Hypothesis: The missing operator is related to the Fine Structure Constant (Alpha).")
    
    # Calculate mean power of alpha
    powers = [math.log(r) / math.log(ALPHA) for r in results.values()]
    mean_p = sum(powers) / len(powers)
    
    print(f"Average Alpha Power: {mean_p:.4f}")
    
    # Test integer/half-integer powers
    candidates = [4, 5, 5.5, 6]
    best_candidate = None
    min_error = 1e9
    
    for p in candidates:
        omega_hyp = ALPHA ** p
        print(f"\nTesting Hypothesis: Omega = Alpha^{p}")
        total_error = 0
        for name in PARTICLES:
            req = results[name]
            err = abs(req - omega_hyp) / req * 100
            print(f"  {name}: Error {err:.2f}%")
            total_error += err
        
        if total_error < min_error:
            min_error = total_error
            best_candidate = p
            
    print(f"\nBest Fit: The geometry implies a coupling of Alpha^{best_candidate}")
    
    if best_candidate == 5:
        print("INTERPRETATION: This corresponds to a 5-dimensional gauge projection (Kaluza-Klein type).")
    elif best_candidate == 6:
        print("INTERPRETATION: This corresponds to a 6-loop self-energy correction.")
        
    print("\nAction: To resolve the gap, we must define the Mass Operator NOT as a simple power,")
    print("but as a composite of Fractal Scale (Phi) and Gauge Coupling (Alpha).")
    print(f"Proposed Formula: m = m_P * phi^-n * alpha^{best_candidate}")

if __name__ == "__main__":
    check_scalings()
