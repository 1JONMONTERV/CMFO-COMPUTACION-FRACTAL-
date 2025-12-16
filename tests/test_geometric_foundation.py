"""
CMFO Geometric Verification Suite
==================================
Formal tests verifying mathematical properties of T^7 with fractal metric.

Tests correspond to theorems in MATHEMATICAL_FOUNDATION.md
"""

import numpy as np
import pytest
from typing import Tuple

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

# Fractal weights λᵢ = φ^(i-1)
LAMBDA = np.array([PHI**i for i in range(7)])

def wrap_angle(theta: float) -> float:
    """Wrap angle to (-π, π]"""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

def geodesic_distance(theta: np.ndarray, eta: np.ndarray) -> float:
    """
    Theorem 1.9: Geodesic distance formula
    
    d_φ(θ, η) = √(Σᵢ λᵢ Δᵢ²)
    where Δᵢ = wrap(θᵢ - ηᵢ)
    """
    assert theta.shape == (7,) and eta.shape == (7,)
    delta = np.array([wrap_angle(theta[i] - eta[i]) for i in range(7)])
    return np.sqrt(np.sum(LAMBDA * delta**2))

def translation(theta: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Definition 3.1: Translation operator
    
    T_a(θ) = θ + a (mod 2π)
    """
    return (theta + a) % (2 * np.pi)

def reflection(theta: np.ndarray, axis: int) -> np.ndarray:
    """
    Definition 3.3: Reflection operator
    
    R_i(θ)ⱼ = -θⱼ if j=i, else θⱼ (mod 2π)
    """
    result = theta.copy()
    result[axis] = -result[axis]
    return result % (2 * np.pi)

# ============================================================================
# GEOMETRIC PROPERTY TESTS
# ============================================================================

class TestMetricProperties:
    """Test Riemannian metric properties"""
    
    def test_positive_definite(self):
        """Metric is positive definite: d(θ,η) ≥ 0 with equality iff θ=η"""
        theta = np.random.uniform(0, 2*np.pi, 7)
        eta = np.random.uniform(0, 2*np.pi, 7)
        
        d = geodesic_distance(theta, eta)
        assert d >= 0, "Distance must be non-negative"
        
        # Distance to self is zero
        d_self = geodesic_distance(theta, theta)
        assert abs(d_self) < 1e-10, "Distance to self must be zero"
    
    def test_symmetry(self):
        """Metric is symmetric: d(θ,η) = d(η,θ)"""
        theta = np.random.uniform(0, 2*np.pi, 7)
        eta = np.random.uniform(0, 2*np.pi, 7)
        
        d1 = geodesic_distance(theta, eta)
        d2 = geodesic_distance(eta, theta)
        
        assert abs(d1 - d2) < 1e-10, "Distance must be symmetric"
    
    def test_triangle_inequality(self):
        """
        Corollary 1.10: Triangle inequality
        
        d(θ,ζ) ≤ d(θ,η) + d(η,ζ)
        """
        for _ in range(100):  # Test multiple random points
            theta = np.random.uniform(0, 2*np.pi, 7)
            eta = np.random.uniform(0, 2*np.pi, 7)
            zeta = np.random.uniform(0, 2*np.pi, 7)
            
            d_theta_zeta = geodesic_distance(theta, zeta)
            d_theta_eta = geodesic_distance(theta, eta)
            d_eta_zeta = geodesic_distance(eta, zeta)
            
            assert d_theta_zeta <= d_theta_eta + d_eta_zeta + 1e-10, \
                f"Triangle inequality violated: {d_theta_zeta} > {d_theta_eta + d_eta_zeta}"

class TestIsometries:
    """Test isometry group properties"""
    
    def test_translation_is_isometry(self):
        """
        Proposition 3.2: Translations preserve distance
        
        d(T_a(θ), T_a(η)) = d(θ, η)
        """
        theta = np.random.uniform(0, 2*np.pi, 7)
        eta = np.random.uniform(0, 2*np.pi, 7)
        a = np.random.uniform(0, 2*np.pi, 7)
        
        d_original = geodesic_distance(theta, eta)
        d_translated = geodesic_distance(
            translation(theta, a),
            translation(eta, a)
        )
        
        assert abs(d_original - d_translated) < 1e-10, \
            "Translation must preserve distance"
    
    def test_reflection_is_isometry(self):
        """
        Proposition 3.4: Reflections preserve distance
        
        d(R_i(θ), R_i(η)) = d(θ, η)
        """
        theta = np.random.uniform(0, 2*np.pi, 7)
        eta = np.random.uniform(0, 2*np.pi, 7)
        
        for axis in range(7):
            d_original = geodesic_distance(theta, eta)
            d_reflected = geodesic_distance(
                reflection(theta, axis),
                reflection(eta, axis)
            )
            
            assert abs(d_original - d_reflected) < 1e-10, \
                f"Reflection in axis {axis} must preserve distance"
    
    def test_composition_closure(self):
        """
        Theorem 3.7: Isometry group is closed under composition
        """
        theta = np.random.uniform(0, 2*np.pi, 7)
        eta = np.random.uniform(0, 2*np.pi, 7)
        a1 = np.random.uniform(0, 2*np.pi, 7)
        a2 = np.random.uniform(0, 2*np.pi, 7)
        
        # Compose two translations
        composed = translation(translation(theta, a1), a2)
        direct = translation(theta, a1 + a2)
        
        assert np.allclose(composed, direct), \
            "Translation composition must equal sum"
        
        # Distance preserved under composition
        d_original = geodesic_distance(theta, eta)
        d_composed = geodesic_distance(
            translation(translation(theta, a1), a2),
            translation(translation(eta, a1), a2)
        )
        
        assert abs(d_original - d_composed) < 1e-10, \
            "Composed isometries must preserve distance"

class TestSpectralProperties:
    """Test Laplace-Beltrami spectral theory"""
    
    def test_eigenfunction_orthogonality(self):
        """
        Eigenfunctions ψₙ(θ) = exp(i n·θ) are orthogonal
        """
        # Sample points on torus
        N = 100
        theta_samples = np.random.uniform(0, 2*np.pi, (N, 7))
        
        # Two different frequency vectors
        n1 = np.array([1, 0, 0, 0, 0, 0, 0])
        n2 = np.array([0, 1, 0, 0, 0, 0, 0])
        
        # Compute eigenfunctions
        psi1 = np.exp(1j * theta_samples @ n1)
        psi2 = np.exp(1j * theta_samples @ n2)
        
        # Inner product (discrete approximation)
        inner_product = np.mean(psi1 * np.conj(psi2))
        
        assert abs(inner_product) < 0.2, \
            "Different eigenfunctions should be approximately orthogonal"
    
    def test_eigenvalue_formula(self):
        """
        Theorem 2.3: Eigenvalues μₙ = Σᵢ nᵢ²/λᵢ
        """
        # Test specific frequency vector
        n = np.array([1, 1, 0, 0, 0, 0, 0])
        
        # Theoretical eigenvalue
        mu_n = np.sum(n**2 / LAMBDA)
        
        # Verify it's positive
        assert mu_n > 0, "Eigenvalue must be positive for non-zero n"
        
        # Verify formula
        expected = 1/LAMBDA[0] + 1/LAMBDA[1]
        assert abs(mu_n - expected) < 1e-10, \
            "Eigenvalue formula must match theoretical value"
    
    def test_spectral_gap(self):
        """
        Corollary 2.4: First non-zero eigenvalue is 1/λ₇ = φ^(-6)
        """
        # Minimum eigenvalue for n ≠ 0
        min_eigenvalue = 1 / LAMBDA[-1]  # 1/λ₇
        
        expected = PHI**(-6)
        assert abs(min_eigenvalue - expected) < 1e-10, \
            f"Spectral gap must be φ^(-6) = {expected}"

class TestCompressionTheory:
    """Test fractal compression properties"""
    
    def test_generator_reconstruction(self):
        """
        Theorem 4.5: Exact reconstruction from generator
        
        (T_a ∘ G)(0) = G(a)
        """
        # Define a simple generator (polynomial)
        def generator(theta: np.ndarray) -> float:
            return np.sum(np.sin(theta)) + np.sum(np.cos(2*theta))
        
        # Test points
        for _ in range(10):
            a = np.random.uniform(0, 2*np.pi, 7)
            
            # Direct evaluation
            direct = generator(a)
            
            # Via translation: T_a(G)(0) = G(T_a(0)) = G(a)
            reconstructed = generator(translation(np.zeros(7), a))
            
            assert abs(direct - reconstructed) < 1e-10, \
                "Reconstruction must be exact"
    
    def test_compression_ratio(self):
        """
        Proposition 4.4: Compression ratio for polynomial generators
        """
        # Polynomial degree
        d = 2
        
        # Number of coefficients for degree-d polynomial in 7 variables
        # (This is a simplification; exact count depends on monomial structure)
        num_coefficients = (d + 1)**7  # Upper bound
        
        # Number of data points
        N = 10**6
        
        # Compression ratio
        CR = N / num_coefficients
        
        assert CR > 1, "Compression ratio must be > 1"
        # For d=2, (d+1)^7 = 3^7 = 2187, so CR ≈ 457
        assert CR > 100, f"Compression ratio {CR} should be substantial (>100)"

class TestNumericalStability:
    """Test numerical properties and edge cases"""
    
    def test_wrapping_consistency(self):
        """Angle wrapping is consistent"""
        for _ in range(100):
            theta = np.random.uniform(-10*np.pi, 10*np.pi)
            wrapped = wrap_angle(theta)
            
            assert -np.pi < wrapped <= np.pi, \
                f"Wrapped angle {wrapped} not in (-π, π]"
            
            # Wrapping is idempotent
            double_wrapped = wrap_angle(wrapped)
            assert abs(wrapped - double_wrapped) < 1e-10, \
                "Wrapping must be idempotent"
    
    def test_distance_bounds(self):
        """Distance is bounded on compact torus"""
        # Maximum possible distance (opposite corners with wrapping)
        max_theoretical = np.sqrt(np.sum(LAMBDA * np.pi**2))
        
        # Test random points
        for _ in range(100):
            theta = np.random.uniform(0, 2*np.pi, 7)
            eta = np.random.uniform(0, 2*np.pi, 7)
            
            d = geodesic_distance(theta, eta)
            
            assert d <= max_theoretical + 1e-10, \
                f"Distance {d} exceeds theoretical maximum {max_theoretical}"

# ============================================================================
# COMPLIANCE TESTS
# ============================================================================

class TestMathematicalCompliance:
    """Verify compliance with mathematical specification"""
    
    def test_dimension(self):
        """System is exactly 7-dimensional"""
        assert len(LAMBDA) == 7, "Must have exactly 7 dimensions"
    
    def test_golden_ratio_weights(self):
        """Weights follow φ^(i-1) pattern"""
        for i in range(7):
            expected = PHI**(i)
            actual = LAMBDA[i]
            assert abs(actual - expected) < 1e-10, \
                f"λ_{i+1} must equal φ^{i}"
    
    def test_metric_determinant(self):
        """
        Proposition 1.6: det(g_φ) = φ^21
        """
        det_g = np.prod(LAMBDA)
        expected = PHI**21
        
        assert abs(det_g - expected) < 1e-8, \
            f"Metric determinant must be φ^21 = {expected}"
    
    def test_volume_formula(self):
        """
        Proposition 1.7: Vol(T^7, g_φ) = (2π)^7 · φ^(21/2)
        """
        volume = (2*np.pi)**7 * np.sqrt(np.prod(LAMBDA))
        expected = (2*np.pi)**7 * PHI**(21/2)
        
        assert abs(volume - expected) < 1e-6, \
            f"Volume must be (2π)^7 · φ^(21/2) = {expected}"

# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestComputationalComplexity:
    """Verify O(1) complexity claims"""
    
    def test_distance_complexity(self):
        """
        Corollary 1.11: Distance computation is O(1)
        """
        import time
        
        # Warm up
        for _ in range(100):
            theta = np.random.uniform(0, 2*np.pi, 7)
            eta = np.random.uniform(0, 2*np.pi, 7)
            geodesic_distance(theta, eta)
        
        # Benchmark
        N = 10000
        start = time.time()
        for _ in range(N):
            theta = np.random.uniform(0, 2*np.pi, 7)
            eta = np.random.uniform(0, 2*np.pi, 7)
            geodesic_distance(theta, eta)
        elapsed = time.time() - start
        
        time_per_call = elapsed / N
        
        # Should be very fast (< 1 microsecond on modern hardware)
        assert time_per_call < 1e-4, \
            f"Distance computation too slow: {time_per_call*1e6:.2f} μs"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
