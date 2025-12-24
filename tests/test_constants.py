"""
Tests for CMFO Constants
=========================
"""

import pytest
import math
from cmfo.constants import PHI, PHI_INV, HBAR, C, G, ALPHA, M_PLANCK


class TestMathematicalConstants:
    """Tests for mathematical constants."""
    
    def test_phi_value(self):
        """Test golden ratio value."""
        expected = (1 + math.sqrt(5)) / 2
        assert abs(PHI - expected) < 1e-15
    
    def test_phi_inv_value(self):
        """Test inverse golden ratio value."""
        assert abs(PHI_INV - 1/PHI) < 1e-15
    
    def test_phi_property(self):
        """Test φ² = φ + 1."""
        assert abs(PHI**2 - (PHI + 1)) < 1e-12
    
    def test_phi_inv_property(self):
        """Test φ⁻¹ = φ - 1."""
        assert abs(PHI_INV - (PHI - 1)) < 1e-12


class TestPhysicalConstants:
    """Tests for physical constants."""
    
    def test_hbar_value(self):
        """Test reduced Planck constant (CODATA 2018)."""
        assert abs(HBAR - 1.054571817e-34) < 1e-42
    
    def test_c_value(self):
        """Test speed of light (exact)."""
        assert C == 299792458.0
    
    def test_g_value(self):
        """Test gravitational constant (CODATA 2018)."""
        assert abs(G - 6.67430e-11) < 1e-16
    
    def test_alpha_value(self):
        """Test fine structure constant."""
        assert abs(ALPHA - 7.29735256e-3) < 1e-11


class TestDerivedConstants:
    """Tests for derived constants."""
    
    def test_planck_mass_value(self):
        """Test Planck mass calculation."""
        expected = math.sqrt((HBAR * C) / G)
        assert abs(M_PLANCK - expected) < 1e-15
    
    def test_planck_mass_order_of_magnitude(self):
        """Test Planck mass is ~10⁻⁸ kg."""
        assert 1e-9 < M_PLANCK < 1e-7
