"""
Tests for CMFO Logic Module
============================
"""

import pytest
from cmfo.logic import TRUE, FALSE, NEUTRAL, f_not, f_and, f_or, f_xor
from cmfo.constants import PHI_INV


class TestLogicConstants:
    """Tests for logic constants."""
    
    def test_true_value(self):
        """Test TRUE constant."""
        assert TRUE == 1.0
    
    def test_false_value(self):
        """Test FALSE constant."""
        assert FALSE == 0.0
    
    def test_neutral_value(self):
        """Test NEUTRAL constant equals φ⁻¹."""
        assert abs(NEUTRAL - PHI_INV) < 1e-12


class TestFractalNot:
    """Tests for fractal NOT operation."""
    
    def test_not_true(self):
        """Test NOT TRUE = FALSE."""
        assert f_not(TRUE) == FALSE
    
    def test_not_false(self):
        """Test NOT FALSE = TRUE."""
        assert f_not(FALSE) == TRUE
    
    def test_not_neutral(self):
        """Test NOT NEUTRAL."""
        result = f_not(NEUTRAL)
        expected = 1.0 - PHI_INV
        assert abs(result - expected) < 1e-12
    
    def test_double_negation(self):
        """Test NOT(NOT x) = x."""
        x = 0.7
        assert abs(f_not(f_not(x)) - x) < 1e-12


class TestFractalAnd:
    """Tests for fractal AND operation."""
    
    def test_and_true_true(self):
        """Test TRUE AND TRUE."""
        result = f_and(TRUE, TRUE)
        expected = TRUE ** PHI_INV
        assert abs(result - expected) < 1e-12
    
    def test_and_true_false(self):
        """Test TRUE AND FALSE = FALSE."""
        assert f_and(TRUE, FALSE) == FALSE
    
    def test_and_false_true(self):
        """Test FALSE AND TRUE = FALSE."""
        assert f_and(FALSE, TRUE) == FALSE
    
    def test_and_false_false(self):
        """Test FALSE AND FALSE = FALSE."""
        assert f_and(FALSE, FALSE) == FALSE
    
    def test_and_commutative(self):
        """Test AND is commutative."""
        a, b = 0.6, 0.8
        assert abs(f_and(a, b) - f_and(b, a)) < 1e-12
    
    def test_and_neutral(self):
        """Test AND with NEUTRAL."""
        result = f_and(NEUTRAL, NEUTRAL)
        expected = (NEUTRAL * NEUTRAL) ** PHI_INV
        assert abs(result - expected) < 1e-12


class TestFractalOr:
    """Tests for fractal OR operation."""
    
    def test_or_true_true(self):
        """Test TRUE OR TRUE."""
        result = f_or(TRUE, TRUE)
        assert abs(result - TRUE) < 1e-10  # Should be close to TRUE
    
    def test_or_true_false(self):
        """Test TRUE OR FALSE."""
        result = f_or(TRUE, FALSE)
        assert abs(result - TRUE) < 1e-10
    
    def test_or_false_true(self):
        """Test FALSE OR TRUE."""
        result = f_or(FALSE, TRUE)
        assert abs(result - TRUE) < 1e-10
    
    def test_or_false_false(self):
        """Test FALSE OR FALSE = FALSE."""
        assert f_or(FALSE, FALSE) == FALSE
    
    def test_or_commutative(self):
        """Test OR is commutative."""
        a, b = 0.6, 0.8
        assert abs(f_or(a, b) - f_or(b, a)) < 1e-12


class TestFractalXor:
    """Tests for fractal XOR operation."""
    
    def test_xor_same_values(self):
        """Test XOR of same values."""
        # XOR should be low for same values
        result = f_xor(TRUE, TRUE)
        assert result < 0.5
        
        result = f_xor(FALSE, FALSE)
        assert result == FALSE
    
    def test_xor_different_values(self):
        """Test XOR of different values."""
        # XOR should be high for different values
        result = f_xor(TRUE, FALSE)
        assert result > 0.5
    
    def test_xor_commutative(self):
        """Test XOR is commutative."""
        a, b = 0.6, 0.8
        assert abs(f_xor(a, b) - f_xor(b, a)) < 1e-12


class TestDeMorganLaws:
    """Tests for De Morgan's laws in fractal logic."""
    
    def test_demorgan_not_and(self):
        """Test NOT(a AND b) ≈ (NOT a) OR (NOT b)."""
        a, b = 0.6, 0.8
        lhs = f_not(f_and(a, b))
        rhs = f_or(f_not(a), f_not(b))
        # May not be exact due to fractal scaling
        assert abs(lhs - rhs) < 0.1
    
    def test_demorgan_not_or(self):
        """Test NOT(a OR b) ≈ (NOT a) AND (NOT b)."""
        a, b = 0.6, 0.8
        lhs = f_not(f_or(a, b))
        rhs = f_and(f_not(a), f_not(b))
        # May not be exact due to fractal scaling
        assert abs(lhs - rhs) < 0.1
