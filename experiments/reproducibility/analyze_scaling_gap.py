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
    print("Searching for the 'Missing Operator' factor...\n")
    
    # Candidates for the missing factor
    candidates = {
        "Delta_Scale (10^12)": 1e12,
        "Alpha^-1 (137)": ALPHA_INV,
        "Alpha^-2": ALPHA_INV**2,
        "Alpha^-4": ALPHA_INV**4,
        "Alpha^-6": ALPHA_INV**6,
        "Phi^12": PHI**12,
        "Phi^24": PHI**24,
        "4*Pi (Spherical)": 4 * PI,
        "Volume (4/3 Pi)": 4/3 * PI
    }

    for name, data in PARTICLES.items():
        n = data['n']
        m_real = data['exp']
        
        # 1. The Raw Fractal Prediction (The failed one)
        m_frac = M_PLANCK_MEV * (PHI ** -n)
        
        # 2. The Ratio (The Error)
        ratio = m_frac / m_real
        
        print(f"=== {name} (n={n}) ===")
        print(f"  Prediction: {m_frac:.4e} MeV")
        print(f"  Real:       {m_real:.4e} MeV")
        print(f"  Gap Ratio:  {ratio:.4e}")
        print(f"  Log10(Gap): {math.log10(ratio):.4f}")
        
        # 3. Pattern Matching
        print("  Potential Matches:")
        found = False
        for cand_name, val in candidates.items():
            # Check if Ratio ~= Candidate * Power
            # Or just check if Ratio ~= Candidate
            if abs(ratio - val) / val < 0.05: # 5% tolerance
                print(f"    [MATCH] ~ {cand_name} (Error: {abs(ratio-val)/val*100:.2f}%)")
                found = True
        
        # Check powers of Alpha specifically
        alpha_power = math.log(ratio) / math.log(ALPHA_INV)
        if abs(round(alpha_power) - alpha_power) < 0.1:
             print(f"    [MATCH] ~ Alpha^-{round(alpha_power)} (Precision: {alpha_power:.3f})")

        if not found and abs(round(alpha_power) - alpha_power) >= 0.1:
             print("    [NO SIMPLE MATCH FOUND]")
        print("")

if __name__ == "__main__":
    check_scalings()
