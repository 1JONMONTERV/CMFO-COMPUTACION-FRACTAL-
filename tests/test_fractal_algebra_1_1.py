"""
Verification Suite for CMFO-FRACTAL-ALGEBRA 1.1
===============================================
"""

import sys
import os
import unittest
import numpy as np

# Add path to bindings
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bindings', 'python'))

from cmfo.core.fractal_algebra_1_1 import (
    FractalUniverse1024, NibbleAlgebra, Renormalization, Metrics, FractalSuite
)

class TestFractalAlgebra1_1(unittest.TestCase):

    def setUp(self):
        # Create random 1024-bit universe (256 nibbles)
        self.data_raw = np.random.randint(0, 16, 256, dtype=np.uint8)
        self.u = FractalUniverse1024(self.data_raw)
        
    def test_01_nibble_invariants(self):
        """1.1 Espejo Niblex Involution: M(M(n)) = n"""
        for n in range(16):
            m = NibbleAlgebra.mirror_4(n)
            m2 = NibbleAlgebra.mirror_4(m)
            self.assertEqual(n, m2, f"Mirror involution failed at {n}")

    def test_02_canon_reconstruction(self):
        """1.2 Reconstruction: n = M^b(C(n))"""
        for n in range(16):
            c, b = NibbleAlgebra.canon_4(n)
            rec = NibbleAlgebra.reconstruct(c, b)
            self.assertEqual(n, rec, f"Reconstruction failed at {n}")
            # Verify property: C(n) <= lex M(C(n))
            # Wait, C(n) is the min. So C(n) <= M(C(n)).
            m_c = NibbleAlgebra.mirror_4(c)
            # In integers, C(n) <= M(C(n))
            # c <= 15 - c => 2c <= 15 => c <= 7. 
            self.assertTrue(c <= 7, f"Canon {c} is not canonical (<=7)")

    def test_03_universe_creation(self):
        """Verify 1024-bit loading"""
        u = FractalUniverse1024(self.data_raw)
        self.assertEqual(len(u.nibbles), 256)
        
    def test_04_global_mirror(self):
        """Global Mirror Involution"""
        m = self.u.mirror()
        m2 = m.mirror()
        np.testing.assert_array_equal(self.u.nibbles, m2.nibbles)
        
    def test_05_canon_global_reconstruction(self):
        """x = M^B(C(x))"""
        c_univ, b_arr = self.u.canon_global()
        rec_univ = c_univ.apply_mirror_mask(b_arr)
        np.testing.assert_array_equal(self.u.nibbles, rec_univ.nibbles)

    def test_06_renorm_commutation(self):
        """rho(Mu, Mv) = M(rho(u,v))"""
        # Testing on raw nibbles first
        for u in range(16):
            for v in range(16):
                mu = NibbleAlgebra.mirror_4(u)
                mv = NibbleAlgebra.mirror_4(v)
                
                # Direct check on Renormalization function
                # Note: The implementation uses u+v//2. 
                # Does (15-u + 15-v)//2 == 15 - (u+v)//2 ?
                rho_uv = Renormalization.rho_sum(u, v)
                rho_mu_mv = Renormalization.rho_sum(mu, mv)
                m_rho = NibbleAlgebra.mirror_4(rho_uv)
                
                self.assertEqual(rho_mu_mv, m_rho, f"Renorm Commutation failed for {u},{v}")

    def test_07_isometry_check(self):
        """Phi(M(x)) = P Phi(x). P=Identity for invariants."""
        diff = Metrics.isometry_check(self.u)
        self.assertLess(diff, 1e-6, "Isometry violation: Metrics changed under mirror")

    def test_08_multiscale_metric(self):
        """d_MS(x,x) = 0"""
        d = Metrics.distance_ms(self.u, self.u)
        self.assertEqual(d, 0.0)
        
        # d(x, M(x)) > 0 usually
        d_m = Metrics.distance_ms(self.u, self.u.mirror())
        # Unless x is symmetric (x = M(x) -> all 7.5? No integers).
        # x=M(x) is impossible for 4-bit integers (n != 15-n ever), so d > 0.
        self.assertGreater(d_m, 0.0)

if __name__ == '__main__':
    unittest.main()
