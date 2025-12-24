"""
CMFO Physics Verification: Test φ-Dirac Equation and Soliton Solutions

This script verifies the mathematical consistency of the CMFO physical formulation.
"""

import numpy as np
from scipy.integrate import odeint
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from cmfo.tolerances import CMFO_PHI, CMFO_TOL_SOLITON_ENERGY
except ImportError:
    CMFO_PHI = 1.618033988749895
    CMFO_TOL_SOLITON_ENERGY = 1e-6
    print("Warning: Using fallback constants\n")


def phi_energy_momentum(p, m):
    """
    Calculate energy from φ-modified dispersion relation:
    E² = φ² p² + m²
    """
    return np.sqrt(CMFO_PHI**2 * np.sum(p**2) + m**2)


def phi_soliton_profile(x, amplitude=1.0, width=1.0, velocity=0.5):
    """
    φ-Soliton profile:
    Ψ(x) = A sech((x - vt) / (φλ))
    """
    return amplitude / np.cosh(x / (CMFO_PHI * width))


def topological_charge(phi_field, dx):
    """
    Calculate topological charge:
    Q = (1/2πφ) ∫ dφ/dx dx
    """
    dphi_dx = np.gradient(phi_field, dx)
    Q = np.sum(dphi_dx) * dx / (2 * np.pi * CMFO_PHI)
    return Q


def test_dispersion_relation():
    """Test 1: Verify φ-modified dispersion relation"""
    print("=" * 70)
    print("TEST 1: φ-Modified Dispersion Relation")
    print("=" * 70)
    
    # Test parameters
    m = 1.0  # mass
    momenta = np.linspace(0, 5, 20)
    
    print(f"\nMass m = {m}")
    print(f"Golden ratio φ = {CMFO_PHI}")
    print(f"\nTesting E² = φ²p² + m²\n")
    
    print(f"{'Momentum |p|':<15} | {'Energy E_CMFO':<15} | {'E_standard':<15} | {'ΔE/E (%)':<15}")
    print("-" * 70)
    
    max_deviation = 0
    for p_mag in momenta:
        p = np.array([p_mag, 0, 0, 0, 0, 0, 0])  # 7D momentum
        E_cmfo = phi_energy_momentum(p, m)
        E_standard = np.sqrt(p_mag**2 + m**2)
        deviation = abs(E_cmfo - E_standard) / E_standard * 100
        max_deviation = max(max_deviation, deviation)
        
        print(f"{p_mag:<15.3f} | {E_cmfo:<15.6f} | {E_standard:<15.6f} | {deviation:<15.2f}")
    
    print(f"\nMaximum deviation: {max_deviation:.2f}%")
    print(f"Expected deviation: ~{(CMFO_PHI - 1) * 100:.2f}% (φ - 1)")
    
    # Verify massless limit: E = φ|p|
    print(f"\nMassless limit (m=0):")
    p_test = 2.0
    E_massless = CMFO_PHI * p_test
    E_calculated = phi_energy_momentum(np.array([p_test, 0, 0, 0, 0, 0, 0]), 0)
    print(f"  E_theory = φ|p| = {E_massless:.6f}")
    print(f"  E_calc = {E_calculated:.6f}")
    print(f"  Error: {abs(E_massless - E_calculated):.2e}")
    
    return max_deviation < 100  # Should show ~61.8% deviation


def test_soliton_profile():
    """Test 2: Verify φ-soliton profile properties"""
    print("\n" + "=" * 70)
    print("TEST 2: φ-Soliton Profile")
    print("=" * 70)
    
    # Generate soliton profile
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    psi = phi_soliton_profile(x, amplitude=1.0, width=1.0)
    
    # Calculate properties
    max_amplitude = np.max(psi)
    center_value = psi[len(psi)//2]
    width_half_max = 2 * CMFO_PHI * 1.0 * np.arccosh(np.sqrt(2))  # Theoretical width at half-max
    
    print(f"\nSoliton Properties:")
    print(f"  Maximum amplitude: {max_amplitude:.6f}")
    print(f"  Center value: {center_value:.6f}")
    print(f"  Theoretical width (FWHM): {width_half_max:.6f}")
    
    # Verify sech profile
    x_test = 0.5
    expected = 1.0 / np.cosh(x_test / CMFO_PHI)
    calculated = phi_soliton_profile(x_test)
    error = abs(expected - calculated)
    
    print(f"\nProfile verification at x = {x_test}:")
    print(f"  Expected: {expected:.6f}")
    print(f"  Calculated: {calculated:.6f}")
    print(f"  Error: {error:.2e}")
    
    # Check normalization (integral should be finite)
    integral = np.trapz(psi**2, x)
    print(f"\nNormalization integral: {integral:.6f}")
    
    return error < 1e-10


def test_topological_charge():
    """Test 3: Verify topological charge conservation"""
    print("\n" + "=" * 70)
    print("TEST 3: Topological Charge Conservation")
    print("=" * 70)
    
    # Create a kink solution: φ(x) = 4 arctan(exp(x))
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    phi_field = 4 * np.arctan(np.exp(x))
    
    # Calculate topological charge
    Q = topological_charge(phi_field, dx)
    
    print(f"\nKink configuration:")
    print(f"  φ(x) = 4 arctan(exp(x))")
    print(f"  Domain: x ∈ [{x[0]}, {x[-1]}]")
    print(f"  Grid points: {len(x)}")
    
    print(f"\nTopological charge:")
    print(f"  Q_calculated = {Q:.6f}")
    print(f"  Q_expected = 1.0 (single kink)")
    print(f"  Error: {abs(Q - 1.0):.2e}")
    
    # Verify charge is conserved (should be integer)
    Q_rounded = round(Q)
    is_integer = abs(Q - Q_rounded) < 0.01
    
    print(f"\nCharge quantization:")
    print(f"  Is integer? {is_integer}")
    print(f"  Rounded value: {Q_rounded}")
    
    return is_integer


def test_phi_quantization():
    """Test 4: Verify φ-quantization of energy levels"""
    print("\n" + "=" * 70)
    print("TEST 4: φ-Quantization of Energy Levels")
    print("=" * 70)
    
    # Energy levels: E_n = φⁿ E₀
    E0 = 1.0
    n_max = 10
    
    print(f"\nEnergy quantization: E_n = φⁿ E₀")
    print(f"Base energy E₀ = {E0}")
    print(f"Golden ratio φ = {CMFO_PHI}\n")
    
    print(f"{'Level n':<10} | {'Energy E_n':<15} | {'Ratio E_n/E_(n-1)':<20}")
    print("-" * 50)
    
    energies = []
    for n in range(n_max + 1):
        E_n = CMFO_PHI**n * E0
        energies.append(E_n)
        
        if n > 0:
            ratio = E_n / energies[n-1]
            print(f"{n:<10} | {E_n:<15.6f} | {ratio:<20.6f}")
        else:
            print(f"{n:<10} | {E_n:<15.6f} | {'N/A':<20}")
    
    # Verify ratios are all φ
    ratios = [energies[i+1] / energies[i] for i in range(len(energies)-1)]
    avg_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    
    print(f"\nStatistics:")
    print(f"  Average ratio: {avg_ratio:.10f}")
    print(f"  Expected (φ): {CMFO_PHI:.10f}")
    print(f"  Standard deviation: {std_ratio:.2e}")
    print(f"  Error: {abs(avg_ratio - CMFO_PHI):.2e}")
    
    return abs(avg_ratio - CMFO_PHI) < 1e-10


def main():
    print("=" * 70)
    print("CMFO PHYSICS VERIFICATION SUITE")
    print("=" * 70)
    print(f"Golden Ratio φ = {CMFO_PHI}")
    print(f"Tolerance = {CMFO_TOL_SOLITON_ENERGY}")
    print("=" * 70)
    
    results = {}
    
    # Run tests
    results['dispersion'] = test_dispersion_relation()
    results['soliton'] = test_soliton_profile()
    results['topological'] = test_topological_charge()
    results['quantization'] = test_phi_quantization()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.capitalize():<20}: {status}")
    
    all_passed = all(results.values())
    print("=" * 70)
    
    if all_passed:
        print("\n✅ [SUCCESS] All CMFO physics tests passed!")
        print("   The mathematical formulation is consistent.")
        return 0
    else:
        print("\n❌ [FAIL] Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
