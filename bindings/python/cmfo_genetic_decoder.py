
import os
import sys
import struct
import math
import numpy as np
import random
import collections

class GeneticDecoder:
    """
    Decodes SHA-256 Nonces as DNA Sequences (ACGT).
    """
    
    def __init__(self):
        self.map_2bit = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        self.codon_map = {
            'A': 0, 'C': 1, 'G': 2, 'T': 3
        }

    def nonce_to_dna(self, nonce):
        # 32 bits = 16 bases
        dna = []
        val = nonce
        for _ in range(16):
            pair = val & 3
            dna.append(self.map_2bit[pair])
            val >>= 2
        return "".join(reversed(dna))

    def analyze_sequence(self, dna):
        length = len(dna)
        
        # 1. GC Content (Stability)
        gc_count = dna.count('G') + dna.count('C')
        gc_content = gc_count / length
        
        # 2. Codon Bias (Entropy of triplets)
        # 16 bases -> 5 codons + 1 base
        codons = [dna[i:i+3] for i in range(0, length-2, 3)]
        codon_counts = collections.Counter(codons)
        # Entropy
        probs = np.array(list(codon_counts.values())) / len(codons)
        bias_entropy = -np.sum(probs * np.log2(probs))
        
        # 3. Palindromes (Hairpins)
        # e.g. AATT (classic palindrome is reverse complement)
        # Simpler definition: "Symmetry"
        # Let's count Palindromic Substrings > 3
        pal_score = 0
        for i in range(length):
            for j in range(i+2, length+1):
                sub = dna[i:j]
                if sub == sub[::-1]:
                    pal_score += len(sub)
        
        # 4. "Start/Stop" framing
        # ATG is Start. TAA, TAG, TGA are Stop.
        has_start = 'ATG' in dna
        has_stop = ('TAA' in dna) or ('TAG' in dna) or ('TGA' in dna)
        gene_framing = 1.0 if (has_start and has_stop) else 0.0
        
        return np.array([gc_content, bias_entropy, pal_score, gene_framing])

    def run_analysis(self):
        print("--- GENETIC CODE DECRYPTION (DNA ANALYSIS) ---")
        
        # Real Nonce: 3536931971
        real_nonce = 3536931971
        dna_real = self.nonce_to_dna(real_nonce)
        print(f"Real Nonce DNA: {dna_real}")
        
        v_real = self.analyze_sequence(dna_real)
        
        print("\n[REAL DNA PHENOTYPE]")
        print(f"GC Content:    {v_real[0]:.4f}")
        print(f"Codon Entropy: {v_real[1]:.4f}")
        print(f"Palindrome Sc: {v_real[2]:.4f}")
        print(f"Gene Framing:  {v_real[3]:.4f}")
        
        # Population
        print("\n[RANDOM MUTATION POOL (N=1000)]")
        pop = []
        for _ in range(1000):
            r = random.randint(0, 2**32-1)
            d = self.nonce_to_dna(r)
            pop.append(self.analyze_sequence(d))
            
        pop = np.array(pop)
        means = np.mean(pop, axis=0)
        stds = np.std(pop, axis=0)
        
        # Avoid div by zero
        stds[stds == 0] = 1e-9
        
        z_scores = (v_real - means) / stds
        
        print("-" * 60)
        names = ["GC Content", "Codon Entropy", "Palindromes", "Gene Framing"]
        max_sig = 0
        best_trait = ""
        
        print(f"{'TRAIT':<15} | {'REAL':<10} | {'POP AVG':<10} | {'SIGMA':<10}")
        print("-" * 60)
        for i in range(4):
            print(f"{names[i]:<15} | {v_real[i]:<10.4f} | {means[i]:<10.4f} | {z_scores[i]:<10.2f}")
            if abs(z_scores[i]) > max_sig:
                max_sig = abs(z_scores[i])
                best_trait = names[i]
        print("-" * 60)
        
        print(f"\nDOMINANT GENETIC TRAIT: {best_trait} ({max_sig:.2f} Sigma)")
        
        if max_sig > 3.0:
            print(">>> GENETIC ANOMALY DETECTED: Nonce is a valid Gene!")
        else:
            print(">>> RESULT: Nonce is Genetic Junk (Non-coding DNA).")

if __name__ == "__main__":
    decoder = GeneticDecoder()
    decoder.run_analysis()
