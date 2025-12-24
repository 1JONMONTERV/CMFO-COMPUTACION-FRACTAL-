"""
CMFO Level 4 CORRECTED: Distance-Based Convergence
===================================================
Implements proper attractor detection using d_φ metric.

Based on spec/metrics.md formal specification.
"""

import sys
import os
import csv
import time
import random
import math

# Add paths
sys.path.insert(0, os.path.abspath('bindings/python'))
sys.path.insert(0, os.path.abspath('.'))

from cmfo.core.structural import FractalVector7
from cortex.encoder import FractalEncoder

PHI = 1.6180339887

class Level4Corrected:
    def __init__(self, vocab_file='vocabulario/spanish_words.txt'):
        self.encoder = FractalEncoder()
        self.words = []
        self.vectors = {}
        self.vocab_file = vocab_file
        
        # Attractors (distance-based)
        self.attractors = []  # List of FractalVector7
        self.attractor_counts = []  # Basin sizes
        
        # Metrics
        self.convergent_count = 0
        self.new_attractor_count = 0
        
    def load_vocabulary(self, sample_size=50000):
        """Load sample for testing"""
        print("[1] Loading vocabulary sample...")
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            all_words = [line.strip() for line in f if line.strip()]
        
        self.words = random.sample(all_words, min(sample_size, len(all_words)))
        
        for word in self.words:
            self.vectors[word] = self.encoder.encode(word)
        print(f"    Loaded {len(self.words)} words")
        
    def d_phi(self, vec1, vec2):
        """Phi-weighted Euclidean distance (spec/metrics.md)"""
        dist_sq = 0.0
        for i in range(7):
            weight = PHI ** i
            diff = vec1.v[i] - vec2.v[i]
            dist_sq += weight * diff * diff
        return math.sqrt(dist_sq)
    
    def find_attractor(self, vec, epsilon=0.15):
        """
        Find nearest attractor within epsilon.
        Returns: (index, distance) or (None, None)
        """
        best_idx = None
        best_dist = float('inf')
        
        for i, attractor in enumerate(self.attractors):
            dist = self.d_phi(vec, attractor)
            if dist < epsilon and dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx is not None and best_dist < epsilon:
            return (best_idx, best_dist)
        return (None, None)
    
    def add_or_merge(self, vec, epsilon=0.15):
        """
        Either merge into existing attractor or create new one.
        Returns: ('convergent', idx) or ('new_attractor', idx)
        """
        idx, dist = self.find_attractor(vec, epsilon)
        
        if idx is not None:
            # Convergent: update attractor (moving average)
            old_count = self.attractor_counts[idx]
            new_count = old_count + 1
            
            # Weighted average
            old_vec = self.attractors[idx]
            new_vec_data = [
                (old_count * old_vec.v[i] + vec.v[i]) / new_count
                for i in range(7)
            ]
            self.attractors[idx] = FractalVector7(new_vec_data)
            self.attractor_counts[idx] = new_count
            
            return ('convergent', idx)
        else:
            # New attractor
            self.attractors.append(vec)
            self.attractor_counts.append(1)
            return ('new_attractor', len(self.attractors) - 1)
    
    def evolve_simple(self, initial_vec, cycles=15):
        """Simple evolution: v_new = v * h with normalization"""
        current = list(initial_vec.v)
        h_params = [PHI ** (i % 7) for i in range(7)]
        
        for cycle in range(cycles):
            # Multiply
            new_state = [current[i] * h_params[i] for i in range(7)]
            
            # Normalize
            norm = sum(x**2 for x in new_state)**0.5
            if norm > 1e-6:
                current = [x / norm for x in new_state]
            else:
                break
        
        return FractalVector7(current)
    
    def find_nearest_word(self, vec):
        """Find nearest word in vocabulary"""
        best_word = None
        best_dist = float('inf')
        
        for word in self.words:
            word_vec = self.vectors[word]
            dist = self.encoder.conceptual_distance(vec, word_vec)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        
        return best_word, best_dist
    
    def run_experiment(self, num_combinations=500, epsilon=0.15, output_file='experiments/level4_corrected.csv'):
        """Run Level 4 with corrected distance-based detection"""
        print(f"\n[LEVEL 4 CORRECTED] Distance-Based Convergence")
        print(f"  Combinations: {num_combinations}")
        print(f"  Epsilon: {epsilon}")
        print("="*60)
        
        results = []
        start_time = time.time()
        
        for i in range(num_combinations):
            # Random pair
            word_a = random.choice(self.words)
            word_b = random.choice(self.words)
            if word_a == word_b:
                continue
            
            vec_a = self.vectors[word_a]
            vec_b = self.vectors[word_b]
            
            # Initial mixing
            initial_state = [(a + b) / 2.0 * PHI for a, b in zip(vec_a.v, vec_b.v)]
            initial_vec = FractalVector7(initial_state)
            
            # Evolve
            final_vec = self.evolve_simple(initial_vec, cycles=15)
            
            # Detect attractor
            result_type, attractor_idx = self.add_or_merge(final_vec, epsilon)
            
            if result_type == 'convergent':
                self.convergent_count += 1
            else:
                self.new_attractor_count += 1
            
            # Find nearest word
            nearest_word, distance = self.find_nearest_word(final_vec)
            
            # Record
            results.append([
                word_a,
                word_b,
                nearest_word,
                f"{distance:.4f}",
                result_type,
                attractor_idx,
                self.attractor_counts[attractor_idx]
            ])
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_combinations - i - 1) / rate if rate > 0 else 0
                print(f"    Progress: {i+1}/{num_combinations} | "
                      f"Attractors: {len(self.attractors)} | "
                      f"Conv: {100*self.convergent_count/(i+1):.1f}% | "
                      f"ETA: {eta:.0f}s")
        
        # Save
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'palabra_A', 'palabra_B', 'palabra_final',
                'distancia', 'tipo', 'attractor_id', 'basin_size'
            ])
            writer.writerows(results)
        
        duration = time.time() - start_time
        
        # Analysis
        conv_pct = 100 * self.convergent_count / num_combinations
        diversity = len(self.attractors) / num_combinations
        
        print("\n" + "="*60)
        print("  RESULTS")
        print("="*60)
        print(f"  Total combinations: {num_combinations}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Rate: {num_combinations/duration:.1f} comb/sec")
        print()
        print("  CONVERGENCE METRICS:")
        print(f"    Convergent:     {self.convergent_count:5d} ({conv_pct:.1f}%)")
        print(f"    New Attractors: {self.new_attractor_count:5d}")
        print(f"    Total Attractors: {len(self.attractors)}")
        print(f"    Diversity: {diversity:.3f}")
        print()
        print("  BASIN SIZE DISTRIBUTION:")
        sorted_basins = sorted(self.attractor_counts, reverse=True)
        print(f"    Largest basin: {sorted_basins[0]} ({100*sorted_basins[0]/num_combinations:.1f}%)")
        print(f"    Top 5 basins: {sorted_basins[:5]}")
        print(f"    Median basin: {sorted_basins[len(sorted_basins)//2]}")
        print()
        print(f"  Saved to: {output_file}")
        print("="*60)
        
        return conv_pct, diversity, len(self.attractors)

def main():
    print("="*60)
    print("  CMFO LEVEL 4 CORRECTED")
    print("  Distance-Based Attractor Detection")
    print("="*60)
    
    engine = Level4Corrected()
    engine.load_vocabulary(sample_size=50000)
    
    # Test with 500 combinations
    conv, div, num_attr = engine.run_experiment(
        num_combinations=500,
        epsilon=0.15,
        output_file='experiments/level4_corrected_500.csv'
    )
    
    print("\n" + "="*60)
    print("  INTERPRETATION")
    print("="*60)
    if conv >= 40:
        print("  [OK] Convergence >= 40%")
        print("  => System shows semantic structure")
        print("  => Ready for larger experiments")
    elif conv >= 20:
        print("  [~] Convergence 20-40%")
        print("  => Weak structure, consider operator adjustment")
    else:
        print("  [!] Convergence < 20%")
        print("  => Still too divergent, needs stronger contraction")
    
    if div < 0.7:
        print(f"  [OK] Diversity < 0.7 ({div:.3f})")
        print("  => Attractors are forming")
    else:
        print(f"  [!] Diversity >= 0.7 ({div:.3f})")
        print("  => Too many unique attractors")
    
    print("="*60)

if __name__ == "__main__":
    main()
