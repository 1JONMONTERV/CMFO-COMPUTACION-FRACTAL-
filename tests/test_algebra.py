"""
Tests for CMFO Algebra Module
==============================
"""

import pytest
import math
from cmfo.constants import PHI, PHI_INV
from cmfo.algebra import fractal_product, fractal_add, fractal_root, iterated_fractal_root


class TestFractalProduct:
    """Tests for fractal product operation."""
    
    def test_fractal_product_basic(self):
        """Test basic fractal product calculation."""
        x, y = 2.0, 3.0
        expected = math.exp(math.log(x) * (math.log(y) / math.log(PHI)))
        result = fractal_product(x, y)
        assert abs(result - expected) < 1e-12
    
    def test_fractal_product_identity(self):
        """Test that fractal_product(x, PHI) behaves correctly."""
        x = 2.0
        result = fractal_product(x, PHI)
        # x^(log_φ(φ)) = x^1 = x
        assert abs(result - x) < 1e-12
    
    def test_fractal_product_phi_squared(self):
        """Test fractal_product(PHI, PHI) = PHI^2."""
        result = fractal_product(PHI, PHI)
        expected = PHI ** 2
        assert abs(result - expected) < 1e-12
    
    def test_fractal_product_negative_raises(self):
        """Test that negative inputs raise ValueError."""
        with pytest.raises(ValueError):
            fractal_product(-1.0, 2.0)
        with pytest.raises(ValueError):
            fractal_product(2.0, -1.0)
    
    def test_fractal_product_zero_raises(self):
        """Test that zero inputs raise ValueError."""
        with pytest.raises(ValueError):
            fractal_product(0.0, 2.0)
        with pytest.raises(ValueError):
            fractal_product(2.0, 0.0)


class TestFractalRoot:
    """Tests for fractal root operation."""
    
    def test_fractal_root_basic(self):
        """Test basic fractal root calculation."""
        x = 2.0
        expected = x ** PHI_INV
        result = fractal_root(x)
        assert abs(result - expected) < 1e-12
    
    def test_fractal_root_identity(self):
        """Test fractal_root(1.0) = 1.0."""
        assert abs(fractal_root(1.0) - 1.0) < 1e-12
    
    def test_fractal_root_phi(self):
        """Test fractal_root(PHI) = PHI^(1/PHI)."""
        result = fractal_root(PHI)
        expected = PHI ** PHI_INV
        assert abs(result - expected) < 1e-12
    
    def test_fractal_root_convergence(self):
        """Test that iterated fractal root converges to 1."""
        x = 100.0
        result = iterated_fractal_root(x, 50)
        assert abs(result - 1.0) < 1e-5
    
    def test_fractal_root_negative_raises(self):
        """Test that negative input raises ValueError."""
        with pytest.raises(ValueError):
            fractal_root(-1.0)
    
    def test_fractal_root_zero_raises(self):
        """Test that zero input raises ValueError."""
        with pytest.raises(ValueError):
            fractal_root(0.0)


class TestIteratedFractalRoot:
    """Tests for iterated fractal root."""
    
    def test_iterated_zero_iterations(self):
        """Test that 0 iterations returns original value."""
        x = 5.0
        assert iterated_fractal_root(x, 0) == x
    
    def test_iterated_one_iteration(self):
        """Test that 1 iteration equals single fractal_root."""
        x = 5.0
        assert abs(iterated_fractal_root(x, 1) - fractal_root(x)) < 1e-12
    
    def test_iterated_convergence_large_x(self):
        """Test convergence for large x."""
        result = iterated_fractal_root(1000.0, 50)
        assert abs(result - 1.0) < 1e-5
    
    def test_iterated_convergence_small_x(self):
        """Test convergence for small x."""
        result = iterated_fractal_root(0.01, 50)
        assert abs(result - 1.0) < 1e-5
    
    def test_iterated_negative_iterations_raises(self):
        """Test that negative iterations raise ValueError."""
        with pytest.raises(ValueError):
            iterated_fractal_root(2.0, -1)


class TestFractalAdd:
    """Tests for fractal addition."""
    
    def test_fractal_add_basic(self):
        """Test that fractal add is standard addition."""
        assert fractal_add(2.0, 3.0) == 5.0
    
    def test_fractal_add_commutative(self):
        """Test commutativity."""
        assert fractal_add(2.0, 3.0) == fractal_add(3.0, 2.0)
    
    def test_fractal_add_identity(self):
        """Test additive identity."""
        x = 5.0
        assert fractal_add(x, 0.0) == x
