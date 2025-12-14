import math

# Constants (CODATA 2018 / Standard Physics)
H_BAR = 1.054571817e-34 # J*s
C = 299792458.0         # m/s
G = 6.67430e-11         # m^3 kg^-1 s^-2
EV_TO_JOULE = 1.602176634e-19

# CMFO Axioms
PHI = (1 + math.sqrt(5)) / 2

def calculate_planck_mass():
    """Returns Planck mass in kg"""
    return math.sqrt((H_BAR * C) / G)

def mass_kg_to_mev(mass_kg):
    """Converts kg to MeV/c^2"""
    joules = mass_kg * (C**2)
    ev = joules / EV_TO_JOULE
    return ev / 1e6

def fractal_mass(n):
    """
    The Core CMFO Axiom: m = m_P * phi^(-n)
    """
    m_p = calculate_planck_mass()
    return m_p * (PHI ** -n)

def verify_claims():
    print("=== CMFO Physics Reproduction Suite ===")
    print(f"Planck Mass (Derived): {calculate_planck_mass():.4e} kg")
    print("-" * 60)
    print(f"{'Particle':<15} | {'Fractal Index (n)':<15} | {'Predicted (MeV)':<15} | {'Exp (MeV)':<15} | {'Error %':<10}")
    print("-" * 60)
    
    # Data from Claims Audit
    # Electron: n=51
    # Muon: n=45
    # Proton: n=39
    
    targets = [
        ("Electron", 51, 0.51099895),
        ("Muon", 45, 105.6583755),
        ("Proton", 39, 938.272088)
    ]
    
    for name, n, experimental in targets:
        # Calculate
        mass_kg = fractal_mass(n)
        mass_mev = mass_kg_to_mev(mass_kg)
        
        # Error
        error = abs(mass_mev - experimental) / experimental * 100
        
        print(f"{name:<15} | {n:<15} | {mass_mev:<15.4e} | {experimental:<15.4f} | {error:<10.0f}%")
    
    print("-" * 60)
    print("Conclusion: The Code is reproducible, but the FORMULA from the text has a large scaling gap (~10^12).")
    print("Action Required: Review the exponent 'n' or the base constant 'm_P' in the theory docs.")
    print("This script is ready to verify the correct formula once updated.")

if __name__ == "__main__":
    verify_claims()
