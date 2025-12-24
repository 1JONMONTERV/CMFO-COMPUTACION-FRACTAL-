"""
CMFO Invariant Tests
====================
Automated verification of spec/laws.md Laws 1-3

L1: Closure
L2: Norm Invariance  
L3: Non-commutativity
"""

import sys
import os
import unittest
import math

sys.path.insert(0, os.path.abspath('.'))

from cmfo.core.metrics import CMFOMetrics, PHI


class TestLaw1Closure(unittest.TestCase):
    """L1: ∀ op ∈ Ω, ∀ x,y ∈ X: op(x,y) ∈ X"""
    
    def test_game_operator_closure(self):
        """game(x,y;θ) must produce valid 7D vector"""
        x = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
        y = [0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.1]
        theta = 0.785  # π/4
        
        # game(x,y;θ) = Γ_φ(cos(θ)·x + sin(θ)·y)
        import math
        result = [
            math.cos(theta) * x[i] + math.sin(theta) * y[i]
            for i in range(7)
        ]
        result = CMFOMetrics.normalize_phi(result)
        
        # Must be 7D
        self.assertEqual(len(result), 7)
        
        # Must be finite
        for val in result:
            self.assertTrue(math.isfinite(val))
    
    def test_normalization_closure(self):
        """Γ_φ must always produce valid vector"""
        test_vectors = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [10.0, -5.0, 3.2, -1.1, 0.5, -0.3, 0.1]
        ]
        
        for vec in test_vectors:
            result = CMFOMetrics.normalize_phi(vec)
            self.assertEqual(len(result), 7)
            for val in result:
                self.assertTrue(math.isfinite(val))


class TestLaw2NormInvariance(unittest.TestCase):
    """L2: ∀ op ∈ Ω, ∀ x,y ∈ X: ||op(x,y)||_φ = 1 ± ε_norm"""
    
    def setUp(self):
        self.epsilon_norm = 0.01  # From spec
    
    def test_normalization_preserves_unit_norm(self):
        """Γ_φ(x) must have ||·||_φ = 1"""
        test_vectors = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [10.0, -5.0, 3.2, -1.1, 0.5, -0.3, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        ]
        
        for vec in test_vectors:
            normalized = CMFOMetrics.normalize_phi(vec)
            norm = CMFOMetrics.phi_norm(normalized)
            
            # Must be 1.0 ± epsilon
            self.assertAlmostEqual(norm, 1.0, delta=self.epsilon_norm,
                                 msg=f"Norm {norm} not within tolerance for {vec}")
    
    def test_game_operator_norm_invariance(self):
        """game(x,y;θ) must preserve unit norm"""
        import math
        
        x = CMFOMetrics.normalize_phi([1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01])
        y = CMFOMetrics.normalize_phi([0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.1])
        
        for theta in [0.0, 0.785, 1.57, 3.14]:  # 0, π/4, π/2, π
            result = [
                math.cos(theta) * x[i] + math.sin(theta) * y[i]
                for i in range(7)
            ]
            result = CMFOMetrics.normalize_phi(result)
            norm = CMFOMetrics.phi_norm(result)
            
            self.assertAlmostEqual(norm, 1.0, delta=self.epsilon_norm,
                                 msg=f"game operator failed norm invariance at θ={theta}")


class TestLaw3NonCommutativity(unittest.TestCase):
    """L3: APP_s(APP_o(v,o),s) ≠_ε APP_s(APP_o(v,s),o)"""
    
    def setUp(self):
        self.epsilon_diff = 0.05  # Minimum difference to be "non-commutative"
    
    def test_word_order_matters(self):
        """
        Simulated test: "Juan ve María" ≠ "María ve Juan"
        
        Using game operator as proxy for APP
        """
        import math
        
        # Simulate: Juan, ve, María as normalized vectors
        juan = CMFOMetrics.normalize_phi([1.0, 0.2, 0.3, 0.1, 0.5, 0.2, 0.1])
        ve = CMFOMetrics.normalize_phi([0.5, 1.0, 0.4, 0.3, 0.2, 0.1, 0.05])
        maria = CMFOMetrics.normalize_phi([0.8, 0.3, 1.0, 0.2, 0.4, 0.3, 0.2])
        
        # Order 1: APP_s(APP_o(ve, maria), juan)
        # Simulate: ve ⊗ maria, then result ⊗ juan
        theta1 = 0.785
        temp1 = [
            math.cos(theta1) * ve[i] + math.sin(theta1) * maria[i]
            for i in range(7)
        ]
        temp1 = CMFOMetrics.normalize_phi(temp1)
        
        result1 = [
            math.cos(theta1) * temp1[i] + math.sin(theta1) * juan[i]
            for i in range(7)
        ]
        result1 = CMFOMetrics.normalize_phi(result1)
        
        # Order 2: APP_s(APP_o(ve, juan), maria)
        temp2 = [
            math.cos(theta1) * ve[i] + math.sin(theta1) * juan[i]
            for i in range(7)
        ]
        temp2 = CMFOMetrics.normalize_phi(temp2)
        
        result2 = [
            math.cos(theta1) * temp2[i] + math.sin(theta1) * maria[i]
            for i in range(7)
        ]
        result2 = CMFOMetrics.normalize_phi(result2)
        
        # Must be different
        distance = CMFOMetrics.d_phi(result1, result2)
        
        self.assertGreater(distance, self.epsilon_diff,
                          msg=f"Word order did not produce distinct results (d={distance})")


if __name__ == '__main__':
    # Run tests
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
