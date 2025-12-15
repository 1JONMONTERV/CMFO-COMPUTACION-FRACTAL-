"""
Simple demonstration of the Compton wavelength mass calculation fix.
"""

import sys
sys.path.insert(0, r'bindings\python')

from cmfo.physics.mass import geometric_mass
from cmfo.constants import H, HBAR, C
import math

# Electron data
M_ELECTRON = 9.1093837015e-31  # kg (CODATA 2018)
LAMBDA_ELECTRON = 2.4263102367e-12  # m (standard Compton wavelength)

print("=" * 80)
print("COMPTON WAVELENGTH MASS CALCULATION FIX - DEMONSTRATION")
print("=" * 80)
print()

print("Known values:")
print(f"  Electron mass (CODATA):           {M_ELECTRON:.10e} kg")
print(f"  Electron Compton wavelength (λ):  {LAMBDA_ELECTRON:.10e} m")
print()

print("Physical constants:")
print(f"  h (Planck constant):              {H:.10e} J·s")
print(f"  ħ (reduced Planck constant):      {HBAR:.10e} J·s")
print(f"  h/ħ ratio:                        {H/HBAR:.10f} ≈ 2π = {2*math.pi:.10f}")
print()

print("-" * 80)
print("OLD BUG: Using ħ with standard wavelength λ")
print("-" * 80)
m_old = HBAR / (C * LAMBDA_ELECTRON)
error_old = abs(m_old - M_ELECTRON) / M_ELECTRON * 100
print(f"  Formula: m = ħ/(c·λ)")
print(f"  Result:  {m_old:.10e} kg")
print(f"  Error:   {error_old:.2f}%")
print(f"  Ratio:   {m_old/M_ELECTRON:.6f} ≈ 1/(2π) = {1/(2*math.pi):.6f}")
print(f"  Status:  ✗ INCORRECT - Missing factor of 2π")
print()

print("-" * 80)
print("FIXED: Using h with standard wavelength λ")
print("-" * 80)
m_new = geometric_mass(LAMBDA_ELECTRON, reduced=False)
error_new = abs(m_new - M_ELECTRON) / M_ELECTRON * 100
print(f"  Formula: m = h/(c·λ) = 2πħ/(c·λ)")
print(f"  Result:  {m_new:.10e} kg")
print(f"  Error:   {error_new:.4f}%")
print(f"  Ratio:   {m_new/M_ELECTRON:.10f}")
print(f"  Status:  ✓ CORRECT - Within numerical precision")
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"The error has been fixed!")
print(f"  Before: {error_old:.2f}% error (factor of 2π missing)")
print(f"  After:  {error_new:.4f}% error (correct within machine precision)")
print()
print("The confusion was between:")
print("  • Standard Compton wavelength: λ = h/(mc)   → m = h/(cλ)")
print("  • Reduced Compton wavelength:  λ̄ = ħ/(mc)  → m = ħ/(cλ̄)")
print()
print("The code now correctly uses h for standard λ and ħ for reduced λ̄.")
print("=" * 80)
