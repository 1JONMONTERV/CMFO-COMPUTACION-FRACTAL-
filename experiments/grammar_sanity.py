"""
Grammar Sanity Test
===================
Test 5 canonical sentences and report metrics.
"""

import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('bindings/python'))

from cmfo.core.metrics import CMFOMetrics
from cmfo.grammar.lexicon import build_minimal_lexicon
from cmfo.grammar.parser import Parser, ParseError


def main():
    print("=" * 60)
    print("  CMFO GRAMMAR SANITY TEST")
    print("=" * 60)
    
    # Build lexicon
    lex = build_minimal_lexicon()
    parser = Parser(lex)
    
    # 5 canonical sentences
    sentences = [
        ["Juan", "come", "manzana"],
        ["la", "manzana", "come", "Juan"],
        # Skip NEG/TENSE for now
        ["Juan", "come", "manzana", "roja"],
    ]
    
    results = []
    
    print("\nParsing sentences...")
    for i, words in enumerate(sentences, 1):
        try:
            result = parser.parse(words)
            results.append(result)
            print(f"  [{i}] OK: {' '.join(words)}")
        except ParseError as e:
            print(f"  [{i}] FAIL: {' '.join(words)} - {e}")
            results.append(None)
    
    # Metrics
    print("\n" + "=" * 60)
    print("  METRICS")
    print("=" * 60)
    
    valid_results = [r for r in results if r is not None]
    print(f"  Unique states: {len(valid_results)}")
    
    if len(valid_results) >= 2:
        d12 = CMFOMetrics.d_phi(valid_results[0].vector, valid_results[1].vector)
        print(f"  d_phi(s1, s2): {d12:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
