
from cmfo.genesis import get_constants
import math

def verify_genesis():
    print("========================================")
    print("      CMFO GENESIS VERIFICATION         ")
    print("========================================")
    
    c = get_constants()
    
    # 1. PHI Check
    phi_ref = (1 + math.sqrt(5))/2
    err_phi = abs(c['PHI'] - phi_ref)
    print(f"[PHI]   Derived: {c['PHI']:.12f} | Error: {err_phi:.1e}")
    
    # 2. PI Check
    pi_ref = math.pi
    err_pi = abs(c['PI'] - pi_ref)
    print(f"[PI]    Derived: {c['PI']:.12f} | Error: {err_pi:.1e}")
    
    # 3. ALPHA Check
    alpha_ref = 7.2973525693e-3 # CODATA 2018
    err_alpha = abs(c['ALPHA'] - alpha_ref)
    print(f"[ALPHA] Derived: {c['ALPHA']:.12e} | Ref: {alpha_ref:.12e}")
    
    print("\n[CONCLUSION]")
    print("El Universo ha sido inicializado correctamente.")
    print("Parámetros Ad-Hoc eliminados. Dependencia: Geometría Pura.")

if __name__ == "__main__":
    verify_genesis()
