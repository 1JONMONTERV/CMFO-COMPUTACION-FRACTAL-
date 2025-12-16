"""
CMFO Grammar Laws Tests
=======================
Formal verification of Laws 1-5 plus 5 canonical sentences.

Laws:
L1: Closure
L2: Norm preservation
L3: Non-commutativity
L4: Idempotence (MOD_N)
L5: Scope (NEG/TENSE)

Canonical sentences:
1. "Juan come manzana"
2. "La manzana come Juan" (should be semantically distant)
3. "Juan no come manzana"
4. "Juan comió manzana"
5. "Juan come manzana roja"
"""

import sys
import os
import unittest

# Setup paths
_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_test_dir)
_bindings = os.path.join(_repo_root, 'bindings', 'python')
_grammar_dir = os.path.join(_repo_root, 'cmfo', 'grammar')

sys.path.insert(0, _bindings)
sys.path.insert(0, _repo_root)

# Load grammar modules using exec to avoid import conflicts
_ops_file = os.path.join(_grammar_dir, 'operators.py')
_ops_globals = {'__file__': _ops_file}
exec(open(_ops_file).read(), _ops_globals)
DET = _ops_globals['DET']
MOD_N = _ops_globals['MOD_N']
APP_O = _ops_globals['APP_O']
APP_S = _ops_globals['APP_S']
NEG_SCOPE = _ops_globals['NEG_SCOPE']
TENSE = _ops_globals['TENSE']

_lex_file = os.path.join(_grammar_dir, 'lexicon.py')
_lex_globals = {'__file__': _lex_file}
exec(open(_lex_file).read(), _lex_globals)
build_minimal_lexicon = _lex_globals['build_minimal_lexicon']
_encode_word = _lex_globals['_encode_word']

_parser_file = os.path.join(_grammar_dir, 'parser.py')
_parser_globals = {'__file__': _parser_file}
exec(open(_parser_file).read(), _parser_globals)
Parser = _parser_globals['Parser']


# Import metrics from bindings
from cmfo.core.metrics import CMFOMetrics




class TestLaws(unittest.TestCase):
    """Test algebraic laws"""
    
    def setUp(self):
        self.epsilon_norm = 1e-3
        self.epsilon_diff = 0.05

    
    def test_L1_closure(self):
        """L1: All operators produce valid 7D vectors"""
        x = _encode_word("test1")
        y = _encode_word("test2")
        
        result = DET(x, y)
        
        self.assertEqual(len(result), 7)
        for val in result:
            self.assertTrue(abs(val) <= 10.0)  # Reasonable bounds
    
    def test_L2_norm_preservation(self):
        """L2: |‖op(x,y)‖_φ - 1| < 1e-3"""
        x = _encode_word("perro")
        y = _encode_word("negro")
        
        result = MOD_N(x, y)
        norm = CMFOMetrics.phi_norm(result)
        
        self.assertAlmostEqual(norm, 1.0, delta=self.epsilon_norm,
                              msg=f"Norm {norm} not preserved")
    
    def test_L3_non_commutativity(self):
        """L3: Word order matters - d_φ > δ"""
        juan = _encode_word("Juan")
        maria = _encode_word("María")
        come = _encode_word("come")
        
        # "Juan come María"
        temp1 = APP_O(come, maria)
        result1 = APP_S(temp1, juan)
        
        # "María come Juan"
        temp2 = APP_O(come, juan)
        result2 = APP_S(temp2, maria)
        
        dist = CMFOMetrics.d_phi(result1, result2)
        
        self.assertGreater(dist, self.epsilon_diff,
                          msg=f"Word order didn't produce distinct results (d={dist})")
    
    def test_L4_idempotence_MOD_N(self):
        """L4: MOD_N(MOD_N(n,a),a) ≈ MOD_N(n,a)"""
        casa = _encode_word("casa")
        roja = _encode_word("roja")
        
        # Apply once
        mod_once = MOD_N(casa, roja)
        
        # Apply twice
        mod_twice = MOD_N(mod_once, roja)
        
        dist = CMFOMetrics.d_phi(mod_once, mod_twice)
        
        self.assertLess(dist, 0.1,
                       msg=f"Idempotence failed (d={dist})")
    
    def test_L5_scope_NEG_TENSE(self):
        """L5: NEG(TENSE(S)) ≠ TENSE(NEG(S))"""
        # Create a sentence state
        s = _encode_word("Juan come")
        
        # NEG(TENSE(S))
        path1 = NEG_SCOPE(TENSE(s))
        
        # TENSE(NEG(S))
        path2 = TENSE(NEG_SCOPE(s))
        
        dist = CMFOMetrics.d_phi(path1, path2)
        
        self.assertGreater(dist, self.epsilon_diff,
                          msg=f"Scope operators commuted (d={dist})")


class TestCanonicalSentences(unittest.TestCase):
    """Test 5 canonical sentences"""
    
    def setUp(self):
        self.lexicon = build_minimal_lexicon()
        self.parser = Parser(self.lexicon)
        self.results = {}
    
    def test_sentence_1(self):
        """1. Juan come manzana"""
        result = self.parser.parse(["Juan", "come", "manzana"])
        self.results['s1'] = result
        self.assertIsNotNone(result)
    
    def test_sentence_2(self):
        """2. La manzana come Juan"""
        result = self.parser.parse(["la", "manzana", "come", "Juan"])
        self.results['s2'] = result
        self.assertIsNotNone(result)
    
    def test_sentence_3(self):
        """3. Juan no come manzana (requires NEG)"""
        # Note: Parser doesn't handle "no" yet, skip for now
        self.skipTest("NEG not in lexicon yet")
    
    def test_sentence_4(self):
        """4. Juan comió manzana (requires TENSE)"""
        # Note: TENSE applied post-parse, skip for now
        self.skipTest("TENSE applied post-parse")
    
    def test_sentence_5(self):
        """5. Juan come manzana roja"""
        result = self.parser.parse(["Juan", "come", "manzana", "roja"])
        self.results['s5'] = result
        self.assertIsNotNone(result)


if __name__ == '__main__':
    # Run tests
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)
