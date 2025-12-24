
import unittest
import sys
sys.path.insert(0, '../bindings/python')
from cmfo.core.structural import FractalVector7

class TestLazyIntegration(unittest.TestCase):

    def test_lazy_detection(self):
        # 1. Basic Vector is Eager
        v1 = FractalVector7()
        self.assertFalse(v1.is_lazy, "Default vector should be Eager")
        
        # 2. Symbolic Vector is Lazy
        v_sym = FractalVector7.symbolic('v')
        self.assertTrue(v_sym.is_lazy, "Symbolic vector should be Lazy")
        
    def test_lazy_propagation(self):
        v_sym = FractalVector7.symbolic('v')
        v_eager = FractalVector7()
        
        # Lazy + Eager -> Lazy
        res = v_sym + v_eager
        self.assertTrue(res.is_lazy, "Lazy + Eager should result in Lazy")
        self.assertIsNotNone(res._node, "Result should have a graph node")
        
        # Check Node Type (String repr check)
        node_str = str(res._node)
        self.assertIn("add", node_str.lower(), "Node operation should be ADD")

    def test_lazy_mul_propagation(self):
        v_sym = FractalVector7.symbolic('v')
        
        # Lazy * Scalar -> Lazy
        res = v_sym * 0.5
        self.assertTrue(res.is_lazy)
        
        # Lazy * Lazy -> Lazy
        h_sym = FractalVector7.symbolic('h')
        res2 = v_sym * h_sym
        self.assertTrue(res2.is_lazy)

if __name__ == '__main__':
    unittest.main()
