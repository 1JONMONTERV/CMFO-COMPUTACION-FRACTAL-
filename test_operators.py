"""
CMFO Level 4 Optimizer: Testing Evolution Operators
====================================================
Tests 3 different operators to find optimal semantic convergence:

A) Contraction: v_new = v * h - α * v²
B) Bounded: v_new = tanh(v * h)  
C) Reduced Cycles: v_new = v * h (but only 5 cycles)

Each tested with 500 combinations.
"""

import sys
import os
import csv
import time
import random
import hashlib
import math

sys.path.insert(0, os.path.abspath('bindings/python'))

from cortex.encoder import FractalEncoder
from cmfo.core.structural import FractalVector7
from cmfo.compiler.jit import FractalJIT

PHI = 1.6180339887

class OperatorTester:
    def __init__(self, vocab_file='vocabulario/spanish_words.txt'):
        self.encoder = FractalEncoder()
        self.words = []
        self.vectors = {}
        self.vocab_file = vocab_file
        
    def load_vocabulary(self, sample_size=50000):
        """Load sample of vocabulary for fast testing"""
        print(f"[*] Loading {sample_size} word sample...")
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            all_words = [line.strip() for line in f if line.strip()]
        
        # Sample for speed
        self.words = random.sample(all_words, min(sample_size, len(all_words)))
        
        # Pre-encode
        for word in self.words:
            self.vectors[word] = self.encoder.encode(word)
        print(f"    Loaded and encoded {len(self.words)} words")
        
    def vector_hash(self, vec):
        vec_str = ','.join(f"{x:.6f}" for x in vec.v)
        return hashlib.md5(vec_str.encode()).hexdigest()[:8]
    
    def find_nearest_word(self, vec):
        best_word = None
        best_dist = float('inf')
        
        for word in self.words:
            word_vec = self.vectors[word]
            dist = self.encoder.conceptual_distance(vec, word_vec)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        
        return best_word, best_dist
    
    def test_operator_A(self, num_tests=500):
        """Operator A: Contraction term"""
        print("\n[TEST A] Contraction: v_new = v*h - 0.1*v²")
        print("="*60)
        
        convergent = 0
        loops = 0
        new_attractors = 0
        attractors = {}
        
        alpha = 0.1  # Contraction strength
        
        for i in range(num_tests):
            word_a = random.choice(self.words)
            word_b = random.choice(self.words)
            if word_a == word_b:
                continue
            
            vec_a = self.vectors[word_a]
            vec_b = self.vectors[word_b]
            
            # Initial mixing
            current = [(a + b) / 2.0 * PHI for a, b in zip(vec_a.v, vec_b.v)]
            
            # Evolve with contraction
            seen_states = set()
            for cycle in range(15):
                state_hash = self.vector_hash(FractalVector7(current))
                
                if state_hash in seen_states:
                    loops += 1
                    break
                seen_states.add(state_hash)
                
                # Apply operator: v_new = v*h - alpha*v²
                h_params = [PHI ** (j % 7) for j in range(7)]
                
                new_state = []
                for j in range(7):
                    linear = current[j] * h_params[j]
                    quadratic = current[j] * current[j]
                    new_state.append(linear - alpha * quadratic)
                
                # Normalize
                norm = sum(x**2 for x in new_state)**0.5
                if norm > 1e-6:
                    current = [x / norm for x in new_state]
                else:
                    break
            
            final_hash = self.vector_hash(FractalVector7(current))
            if final_hash in attractors:
                convergent += 1
            else:
                attractors[final_hash] = True
                new_attractors += 1
            
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{num_tests}")
        
        conv_pct = 100 * convergent / num_tests
        loop_pct = 100 * loops / num_tests
        
        print(f"\n  Results:")
        print(f"    Convergent: {convergent} ({conv_pct:.1f}%)")
        print(f"    Loops: {loops} ({loop_pct:.1f}%)")
        print(f"    Unique attractors: {len(attractors)}")
        
        return conv_pct, loop_pct, len(attractors)
    
    def test_operator_B(self, num_tests=500):
        """Operator B: Bounded activation"""
        print("\n[TEST B] Bounded: v_new = tanh(v*h)")
        print("="*60)
        
        convergent = 0
        loops = 0
        new_attractors = 0
        attractors = {}
        
        for i in range(num_tests):
            word_a = random.choice(self.words)
            word_b = random.choice(self.words)
            if word_a == word_b:
                continue
            
            vec_a = self.vectors[word_a]
            vec_b = self.vectors[word_b]
            
            current = [(a + b) / 2.0 * PHI for a, b in zip(vec_a.v, vec_b.v)]
            
            seen_states = set()
            for cycle in range(15):
                state_hash = self.vector_hash(FractalVector7(current))
                
                if state_hash in seen_states:
                    loops += 1
                    break
                seen_states.add(state_hash)
                
                # Apply operator: v_new = tanh(v*h)
                h_params = [PHI ** (j % 7) for j in range(7)]
                
                new_state = []
                for j in range(7):
                    product = current[j] * h_params[j]
                    new_state.append(math.tanh(product))
                
                current = new_state
            
            final_hash = self.vector_hash(FractalVector7(current))
            if final_hash in attractors:
                convergent += 1
            else:
                attractors[final_hash] = True
                new_attractors += 1
            
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{num_tests}")
        
        conv_pct = 100 * convergent / num_tests
        loop_pct = 100 * loops / num_tests
        
        print(f"\n  Results:")
        print(f"    Convergent: {convergent} ({conv_pct:.1f}%)")
        print(f"    Loops: {loops} ({loop_pct:.1f}%)")
        print(f"    Unique attractors: {len(attractors)}")
        
        return conv_pct, loop_pct, len(attractors)
    
    def test_operator_C(self, num_tests=500):
        """Operator C: Reduced cycles"""
        print("\n[TEST C] Reduced Cycles: v_new = v*h (5 cycles)")
        print("="*60)
        
        convergent = 0
        loops = 0
        new_attractors = 0
        attractors = {}
        
        for i in range(num_tests):
            word_a = random.choice(self.words)
            word_b = random.choice(self.words)
            if word_a == word_b:
                continue
            
            vec_a = self.vectors[word_a]
            vec_b = self.vectors[word_b]
            
            current = [(a + b) / 2.0 * PHI for a, b in zip(vec_a.v, vec_b.v)]
            
            seen_states = set()
            for cycle in range(5):  # Only 5 cycles
                state_hash = self.vector_hash(FractalVector7(current))
                
                if state_hash in seen_states:
                    loops += 1
                    break
                seen_states.add(state_hash)
                
                # Apply operator: v_new = v*h
                h_params = [PHI ** (j % 7) for j in range(7)]
                
                new_state = []
                for j in range(7):
                    new_state.append(current[j] * h_params[j])
                
                # Normalize
                norm = sum(x**2 for x in new_state)**0.5
                if norm > 1e-6:
                    current = [x / norm for x in new_state]
                else:
                    break
            
            final_hash = self.vector_hash(FractalVector7(current))
            if final_hash in attractors:
                convergent += 1
            else:
                attractors[final_hash] = True
                new_attractors += 1
            
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{num_tests}")
        
        conv_pct = 100 * convergent / num_tests
        loop_pct = 100 * loops / num_tests
        
        print(f"\n  Results:")
        print(f"    Convergent: {convergent} ({conv_pct:.1f}%)")
        print(f"    Loops: {loops} ({loop_pct:.1f}%)")
        print(f"    Unique attractors: {len(attractors)}")
        
        return conv_pct, loop_pct, len(attractors)

def main():
    print("="*60)
    print("  CMFO OPERATOR OPTIMIZATION")
    print("  Testing 3 Evolution Operators (500 tests each)")
    print("="*60)
    
    tester = OperatorTester()
    tester.load_vocabulary(sample_size=50000)
    
    results = {}
    
    # Test all three
    conv_a, loop_a, attr_a = tester.test_operator_A()
    results['A'] = (conv_a, loop_a, attr_a)
    
    conv_b, loop_b, attr_b = tester.test_operator_B()
    results['B'] = (conv_b, loop_b, attr_b)
    
    conv_c, loop_c, attr_c = tester.test_operator_C()
    results['C'] = (conv_c, loop_c, attr_c)
    
    # Summary
    print("\n" + "="*60)
    print("  COMPARATIVE SUMMARY")
    print("="*60)
    print(f"  Operator | Convergence | Loops | Attractors")
    print(f"  ---------|-------------|-------|------------")
    print(f"  A (v*h-αv²) | {conv_a:6.1f}%    | {loop_a:4.1f}% | {attr_a:5d}")
    print(f"  B (tanh)    | {conv_b:6.1f}%    | {loop_b:4.1f}% | {attr_b:5d}")
    print(f"  C (5 cyc)   | {conv_c:6.1f}%    | {loop_c:4.1f}% | {attr_c:5d}")
    print("="*60)
    
    # Recommendation
    best = max(results.items(), key=lambda x: x[1][0])  # Max convergence
    print(f"\n  RECOMMENDATION: Operator {best[0]}")
    print(f"    Convergence: {best[1][0]:.1f}%")
    print(f"    Ready for 5k run: {'YES' if best[1][0] >= 40 else 'NO'}")
    print("="*60)

if __name__ == "__main__":
    main()
