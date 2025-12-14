"""
Tests for CMFO Physics Module
==============================
"""

import pytest
from cmfo.physics import geometric_mass, compton_wavelength
from cmfo.constants import HBAR, C


class TestGeometricMass:
    """Tests for geometric mass calculation."""
    
    def test_geometric_mass_basic(self):
        """Test basic geometric mass calculation."""
        L = 1e-12  # 1 picometer
        expected = HBAR / (C * L)
        result = geometric_mass(L)
        assert abs(result - expected) < 1e-50
    
    def test_geometric_mass_electron_scale(self):
        """Test geometric mass at electron Compton wavelength scale."""
        # Electron Compton wavelength
        L_e = 2.4263102367e-12  # meters
        m_e = geometric_mass(L_e)
        # Electron mass
        m_e_expected = 9.1093837015e-31  # kg
        # Should be within 1% (exact in natural units)
        relative_error = abs(m_e - m_e_expected) / m_e_expected
        assert relative_error < 0.01
    
    def test_geometric_mass_negative_raises(self):
        """Test that negative length raises ValueError."""
        with pytest.raises(ValueError):
            geometric_mass(-1e-12)
    
    def test_geometric_mass_zero_raises(self):
        """Test that zero length raises ValueError."""
        with pytest.raises(ValueError):
            geometric_mass(0.0)
    
    def test_geometric_mass_dimensional_analysis(self):
        """Test dimensional correctness."""
        L = 1.0  # 1 meter
        m = geometric_mass(L)
        # Result should have units of kg
        # HBAR [J·s] / (C [m/s] * L [m]) = [kg·m²/s²·s] / [m²/s] = [kg]
        assert m > 0  # Just check it's positive and computable


class TestComptonWavelength:
    """Tests for Compton wavelength calculation."""
    
    def test_compton_wavelength_basic(self):
        """Test basic Compton wavelength calculation."""
        m = 1e-30  # kg
        expected = HBAR / (m * C)
        result = compton_wavelength(m)
        assert abs(result - expected) < 1e-50
    
    def test_compton_wavelength_electron(self):
        """Test Compton wavelength for electron mass."""
        m_e = 9.1093837015e-31  # kg
        lambda_e = compton_wavelength(m_e)
        lambda_e_expected = 2.4263102367e-12  # meters
        relative_error = abs(lambda_e - lambda_e_expected) / lambda_e_expected
        assert relative_error < 0.01
    
    def test_compton_wavelength_negative_raises(self):
        """Test that negative mass raises ValueError."""
        with pytest.raises(ValueError):
            compton_wavelength(-1e-30)
    
    def test_compton_wavelength_zero_raises(self):
        """Test that zero mass raises ValueError."""
        with pytest.raises(ValueError):
            compton_wavelength(0.0)


class TestInverseRelations:
    """Tests for inverse relations between mass and wavelength."""
    
    def test_mass_wavelength_inverse(self):
        """Test that geometric_mass and compton_wavelength are inverses."""
        L = 1e-12
        m = geometric_mass(L)
        L_recovered = compton_wavelength(m)
        assert abs(L - L_recovered) / L < 1e-10
    
    def test_wavelength_mass_inverse(self):
        """Test inverse relation starting from mass."""
        m = 1e-30
        L = compton_wavelength(m)
        m_recovered = geometric_mass(L)
        assert abs(m - m_recovered) / m < 1e-10
