"""
Verification of the Compton wavelength mass calculation fix.

This script demonstrates that the error has been corrected:
- Before: Used ħ with standard λ → gave mass × (1/2π) ≈ 15.9% of real value
- After: Uses h with standard λ → gives correct mass (100%)
"""

import math
import sys
sys.path.insert(0, r'c:\Users\solmo\Desktop\CMFO_GPU_CLEAN\CMFO_GPU_FINAL\bindings\python')

from cmfo.physics.mass import geometric_mass, compton_wavelength
from cmfo.constants import H, HBAR, C

# Known electron data (CODATA 2018)
M_ELECTRON = 9.1093837015e-31  # kg
LAMBDA_ELECTRON = 2.4263102367e-12  # m (standard Compton wavelength)
LAMBDA_ELECTRON_BAR = LAMBDA_ELECTRON / (2 * math.pi)  # reduced wavelength

print("=" * 70)
print("COMPTON WAVELENGTH MASS CALCULATION - VERIFICATION")
print("=" * 70)
print()

# Test 1: Standard wavelength with h (CORRECTED)
print("TEST 1: Standard Compton wavelength (λ) with h")
print("-" * 70)
print(f"Input: λ_e = {LAMBDA_ELECTRON:.10e} m (standard)")
m_calculated = geometric_mass(LAMBDA_ELECTRON, reduced=False)
print(f"Calculated mass: {m_calculated:.10e} kg")
print(f"Expected mass:   {M_ELECTRON:.10e} kg")
error_percent = abs(m_calculated - M_ELECTRON) / M_ELECTRON * 100
print(f"Error: {error_percent:.4f}%")
print(f"✓ PASS" if error_percent < 1.0 else "✗ FAIL")
print()

# Test 2: Reduced wavelength with ħ (should also work)
print("TEST 2: Reduced Compton wavelength (λ̄) with ħ")
print("-" * 70)
print(f"Input: λ̄_e = {LAMBDA_ELECTRON_BAR:.10e} m (reduced)")
m_calculated_bar = geometric_mass(LAMBDA_ELECTRON_BAR, reduced=True)
print(f"Calculated mass: {m_calculated_bar:.10e} kg")
print(f"Expected mass:   {M_ELECTRON:.10e} kg")
error_percent_bar = abs(m_calculated_bar - M_ELECTRON) / M_ELECTRON * 100
print(f"Error: {error_percent_bar:.4f}%")
print(f"✓ PASS" if error_percent_bar < 1.0 else "✗ FAIL")
print()

# Test 3: Verify the old bug would have given 1/(2π) error
print("TEST 3: Demonstrating the OLD BUG (using ħ with standard λ)")
print("-" * 70)
m_old_bug = HBAR / (C * LAMBDA_ELECTRON)  # Old incorrect formula
print(f"Old formula result: {m_old_bug:.10e} kg")
print(f"Expected mass:      {M_ELECTRON:.10e} kg")
bug_error = abs(m_old_bug - M_ELECTRON) / M_ELECTRON * 100
print(f"Error: {bug_error:.4f}%")
print(f"Expected error: ~{(1 - 1/(2*math.pi))*100:.2f}% (missing factor of 2π)")
print(f"Ratio: {m_old_bug/M_ELECTRON:.6f} ≈ 1/(2π) = {1/(2*math.pi):.6f}")
print()

# Test 4: Inverse function (mass → wavelength)
print("TEST 4: Inverse function - mass to wavelength")
print("-" * 70)
lambda_calc = compton_wavelength(M_ELECTRON, reduced=False)
print(f"Calculated λ: {lambda_calc:.10e} m")
print(f"Expected λ:   {LAMBDA_ELECTRON:.10e} m")
lambda_error = abs(lambda_calc - LAMBDA_ELECTRON) / LAMBDA_ELECTRON * 100
print(f"Error: {lambda_error:.4f}%")
print(f"✓ PASS" if lambda_error < 1.0 else "✗ FAIL")
print()

# Test 5: Verify h = 2π·ħ relationship
print("TEST 5: Verify h = 2π·ħ relationship")
print("-" * 70)
h_from_hbar = 2 * math.pi * HBAR
print(f"h (constant):    {H:.10e} J·s")
print(f"2π·ħ (computed): {h_from_hbar:.10e} J·s")
h_error = abs(H - h_from_hbar) / H * 100
print(f"Difference: {h_error:.6f}%")
print(f"✓ PASS" if h_error < 0.01 else "✗ FAIL")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
all_pass = all([
    error_percent < 1.0,
    error_percent_bar < 1.0,
    lambda_error < 1.0,
    h_error < 0.01
])
if all_pass:
    print("✓ ALL TESTS PASSED - Physics is now correct!")
    print(f"  • Standard wavelength calculation: {error_percent:.4f}% error")
    print(f"  • Reduced wavelength calculation: {error_percent_bar:.4f}% error")
    print(f"  • Inverse calculation: {lambda_error:.4f}% error")
else:
    print("✗ SOME TESTS FAILED - Review needed")
print()
print("The bug has been fixed:")
print(f"  Before: {bug_error:.2f}% error (used ħ instead of h)")
print(f"  After:  {error_percent:.4f}% error (now using h correctly)")
print("=" * 70)
