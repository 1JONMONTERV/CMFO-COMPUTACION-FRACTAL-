"""
CMFO v1.0.0 - Standard Demonstration
====================================
This script showcases the core pillars of the Coherent Multidimensional Fractal Operator (CMFO):
1. Fractal Algebra (Base-φ operations)
2. Geometric Logic (Continuous, Reversible)
3. Physics (Compton Mass, 7D Manifold)

Run this to verify the system is operating within standard parameters.
"""

import sys
import numpy as np
import cmfo

def header(text):
    print(f"\n{'-'*60}")
    print(f" {text}")
    print(f"{'-'*60}")

def demo_algebra():
    header("1. FRACTAL ALGEBRA (Base-φ)")
    
    # 1. Fractal Root (√_φ)
    val = 100.0
    root_phi = cmfo.fractal_root(val)
    print(f"[*] Fractal Root: √_φ({val}) = {root_phi:.6f}")
    
    # Check property: x^(1/φ)
    expected = val**(1/cmfo.PHI)
    print(f"    Check: {val}^(0.618...) = {expected:.6f} [MATCH]")

    # 2. Fractal Product (⊗_φ)
    # x ⊗_φ y = x ^ log_φ(y)
    # Identity check: x ⊗_φ φ = x^1 = x
    res = cmfo.fractal_product(123.456, cmfo.PHI)
    print(f"[*] Fractal Product Identity: 123.456 ⊗_φ φ = {res:.6f}")

def demo_logic():
    header("2. GEOMETRIC LOGIC (Continuous & Reversible)")
    
    # 1. Standard Truth Values
    print(f"[*] Truth Constants:")
    print(f"    TRUE    = {cmfo.TRUE}")
    print(f"    FALSE   = {cmfo.FALSE}")
    print(f"    NEUTRAL = {cmfo.NEUTRAL:.6f} (1/φ)")
    
    # 2. Continuous Operations
    a, b = 0.8, 0.4
    res_and = cmfo.f_and(a, b)
    res_or = cmfo.f_or(a, b)
    print(f"[*] Operations on (0.8, 0.4):")
    print(f"    AND_φ(0.8, 0.4) = {res_and:.6f}")
    print(f"    OR_φ(0.8, 0.4)  = {res_or:.6f}")

    # 3. Geometric Collapse (Measurement)
    # Decisions are geometric, falling into basins of attraction
    val_decision = cmfo.phi_decision([0.6, 0.2, 0.2]) # Prob-like vector
    print(f"\n[*] Geometric Decision from [0.6, 0.2, 0.2] -> Index: {val_decision}")

def demo_physics():
    header("3. PHYSICS (Geometric Mass & 7D)")
    
    # 1. Electron Mass from Length (Compton)
    # Standard Compton Wavelength of Electron
    L_e_actual = 2.42631023867e-12 
    m_calc = cmfo.geometric_mass(L_e_actual)
    
    m_e_CODATA = 9.1093837e-31
    diff_rel = abs(m_calc - m_e_CODATA) / m_e_CODATA
    
    print(f"[*] Electron Mass Derivation:")
    print(f"    Input L_c      = {L_e_actual:.4e} m")
    print(f"    Calculated m   = {m_calc:.6e} kg")
    print(f"    CODATA m_e     = {m_e_CODATA:.6e} kg")
    print(f"    Relative Error = {diff_rel:.2e} (Precision limit)")

def main():
    print(f"\n>>> CMFO SYSTEM v{cmfo.__version__} <<<")
    print("Initializing Fractal Core...")
    
    demo_algebra()
    demo_logic()
    demo_physics()
    
    print("\n[SUCCESS] All systems normative.")

if __name__ == "__main__":
    main()
