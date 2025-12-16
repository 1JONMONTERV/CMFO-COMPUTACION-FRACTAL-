"""
CMFO Level 4: Emergent Semantic Dynamics
=========================================
Discovers emergent conceptual relationships through fractal evolution.

Invariants:
- Phi-norm conservation per cycle
- No dimensional collapse
- Minimum distance between distinct states

Metrics per combination:
- palabra_A, palabra_B
- vector_inicial (hash)
- trayectoria (summary)
- palabra_final_mas_cercana
- distancia_geometrica
- tipo_resultado: {convergente, loop, colapso, nuevo_atractor}
"""

import sys
import os
import csv
import time
import random
import hashlib
from collections import defaultdict

sys.path.insert(0, os.path.abspath('bindings/python'))

from cortex.encoder import FractalEncoder
from cmfo.core.structural import FractalVector7
from cmfo.compiler.jit import FractalJIT

PHI = 1.6180339887

class Level4SemanticDynamics:
    def __init__(self, vocab_file='vocabulario/spanish_words.txt'):
        self.encoder = FractalEncoder()
        self.words = []
        self.vectors = {}
        self.vocab_file = vocab_file
        
        # Fractal memory (persistent states)
        self.attractor_memory = {}
        self.trajectory_cache = {}
        
        # Metrics
        self.convergent_count = 0
        self.loop_count = 0
        self.collapse_count = 0
        self.new_attractor_count = 0
        
    def load_vocabulary(self):
        """Load Spanish words"""
        print("[1] Loading Spanish Vocabulary...")
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            self.words = [line.strip() for line in f if line.strip()]
        print(f"    Loaded {len(self.words)} words")
        
        # Pre-encode all words for fast lookup
        print("[2] Pre-encoding vocabulary to 7D space...")
        for i, word in enumerate(self.words):
            self.vectors[word] = self.encoder.encode(word)
            if (i + 1) % 100000 == 0:
                print(f"    Progress: {i+1}/{len(self.words)}")
        print("    Complete.")
        
    def vector_hash(self, vec):
        """Generate deterministic hash of vector state"""
        vec_str = ','.join(f"{x:.6f}" for x in vec.v)
        return hashlib.md5(vec_str.encode()).hexdigest()[:8]
    
    def phi_norm(self, vec):
        """Calculate Phi-weighted norm"""
        return sum(abs(vec.v[i]) * (PHI ** (i % 7)) for i in range(7))
    
    def evolve_with_tracking(self, initial_vec, cycles=15):
        """
        Evolve vector through fractal dynamics with full tracking.
        Returns: (final_vec, trajectory_summary, convergence_type)
        """
        v_sym = FractalVector7.symbolic('v')
        h_sym = FractalVector7.symbolic('h')
        
        # Physics: v_new = v * h (interaction)
        eq = v_sym * h_sym
        
        current_state = list(initial_vec.v)
        h_params = [PHI ** (i % 7) for i in range(7)]
        
        trajectory = []
        state_hashes = set()
        
        initial_norm = self.phi_norm(initial_vec)
        
        for cycle in range(cycles):
            # Track state
            state_hash = self.vector_hash(FractalVector7(current_state))
            trajectory.append(state_hash)
            
            # Detect loop
            if state_hash in state_hashes:
                return (FractalVector7(current_state), 
                       f"loop@{cycle}", 
                       "loop")
            state_hashes.add(state_hash)
            
            # Evolve
            res = FractalJIT.compile_and_run(eq._node, current_state, h_params)
            current_state = [x for row in res for x in row]
            
            # Check invariants
            current_vec = FractalVector7(current_state)
            current_norm = self.phi_norm(current_vec)
            
            # Norm conservation check (within 10%)
            if abs(current_norm - initial_norm) / initial_norm > 0.1:
                # Renormalize to preserve structure
                scale = initial_norm / current_norm if current_norm > 1e-12 else 1.0
                current_state = [x * scale for x in current_state]
            
            # Collapse detection
            if current_norm < 1e-6:
                return (FractalVector7(current_state),
                       f"collapse@{cycle}",
                       "colapso")
        
        # Check convergence
        final_vec = FractalVector7(current_state)
        final_hash = self.vector_hash(final_vec)
        
        # New attractor?
        if final_hash not in self.attractor_memory:
            self.attractor_memory[final_hash] = final_vec
            conv_type = "nuevo_atractor"
        else:
            conv_type = "convergente"
        
        traj_summary = f"{trajectory[0]}..{final_hash}"
        return (final_vec, traj_summary, conv_type)
    
    def find_nearest_word(self, vec):
        """Find nearest word in vocabulary"""
        best_word = None
        best_dist = float('inf')
        
        # Sample search for speed (check 10k random words)
        sample_words = random.sample(self.words, min(10000, len(self.words)))
        
        for word in sample_words:
            word_vec = self.vectors[word]
            dist = self.encoder.conceptual_distance(vec, word_vec)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        
        return best_word, best_dist
    
    def generate_level4(self, num_combinations=5000, output_file='vocabulario/spanish_emergent_L4.csv'):
        """Generate Level 4: Emergent Semantic Dynamics"""
        print(f"\n[LEVEL 4] Emergent Semantic Dynamics ({num_combinations} combinations)")
        print("="*60)
        
        results = []
        start_time = time.time()
        
        for i in range(num_combinations):
            # Select random pair
            word_a = random.choice(self.words)
            word_b = random.choice(self.words)
            
            if word_a == word_b:
                continue
            
            # Initial state: geometric mixing
            vec_a = self.vectors[word_a]
            vec_b = self.vectors[word_b]
            
            initial_state = [(a + b) / 2.0 * PHI for a, b in zip(vec_a.v, vec_b.v)]
            initial_vec = FractalVector7(initial_state)
            initial_hash = self.vector_hash(initial_vec)
            
            # Evolve with tracking
            final_vec, trajectory, conv_type = self.evolve_with_tracking(initial_vec)
            
            # Find nearest word
            nearest_word, distance = self.find_nearest_word(final_vec)
            
            # Classify
            if conv_type == "loop":
                self.loop_count += 1
            elif conv_type == "colapso":
                self.collapse_count += 1
            elif conv_type == "nuevo_atractor":
                self.new_attractor_count += 1
            else:
                self.convergent_count += 1
            
            # Record
            results.append([
                word_a,
                word_b,
                initial_hash,
                trajectory,
                nearest_word,
                f"{distance:.4f}",
                conv_type
            ])
            
            # Progress
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_combinations - i - 1) / rate if rate > 0 else 0
                print(f"    Progress: {i+1}/{num_combinations} | "
                      f"Rate: {rate:.1f} comb/sec | "
                      f"ETA: {eta/60:.1f} min")
        
        # Save results
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'palabra_A', 'palabra_B', 'vector_inicial_hash',
                'trayectoria', 'palabra_final_mas_cercana',
                'distancia_geometrica', 'tipo_resultado'
            ])
            writer.writerows(results)
        
        duration = time.time() - start_time
        
        # Final metrics
        print("\n" + "="*60)
        print("  LEVEL 4 COMPLETE: Emergent Dynamics Analysis")
        print("="*60)
        print(f"  Total combinations: {num_combinations}")
        print(f"  Duration: {duration/60:.1f} minutes")
        print(f"  Rate: {num_combinations/duration:.1f} combinations/sec")
        print()
        print("  CONVERGENCE METRICS:")
        print(f"    Convergent:     {self.convergent_count:5d} ({100*self.convergent_count/num_combinations:.1f}%)")
        print(f"    Loops:          {self.loop_count:5d} ({100*self.loop_count/num_combinations:.1f}%)")
        print(f"    Collapses:      {self.collapse_count:5d} ({100*self.collapse_count/num_combinations:.1f}%)")
        print(f"    New Attractors: {self.new_attractor_count:5d} ({100*self.new_attractor_count/num_combinations:.1f}%)")
        print()
        print(f"  Unique attractors discovered: {len(self.attractor_memory)}")
        print(f"  Saved to: {output_file}")
        print("="*60)
        
        # Scaling recommendation
        conv_pct = 100 * self.convergent_count / num_combinations
        loop_pct = 100 * self.loop_count / num_combinations
        
        print("\n  SCALING RECOMMENDATION:")
        criteria_met = 0
        if conv_pct >= 60:
            print(f"    [OK] Convergence >= 60% ({conv_pct:.1f}%)")
            criteria_met += 1
        else:
            print(f"    [!]  Convergence < 60% ({conv_pct:.1f}%)")
        
        if loop_pct <= 5:
            print(f"    [OK] Loops <= 5% ({loop_pct:.1f}%)")
            criteria_met += 1
        else:
            print(f"    [!]  Loops > 5% ({loop_pct:.1f}%)")
        
        if len(self.attractor_memory) > num_combinations * 0.1:
            print(f"    [OK] Diverse attractors ({len(self.attractor_memory)})")
            criteria_met += 1
        else:
            print(f"    [!]  Low attractor diversity ({len(self.attractor_memory)})")
        
        print()
        if criteria_met >= 3:
            print("    => READY TO SCALE to 10k/50k/100k")
        else:
            print("    => ADJUST parameters before scaling")
        print("="*60)

def main():
    print("="*60)
    print("  CMFO LEVEL 4: EMERGENT SEMANTIC DYNAMICS")
    print("  Spanish Language - Fractal Evolution Experiment")
    print("="*60)
    
    engine = Level4SemanticDynamics()
    engine.load_vocabulary()
    engine.generate_level4(num_combinations=5000)

if __name__ == "__main__":
    main()
