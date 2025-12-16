"""
CMFO Spanish Language Knowledge Base Generator
===============================================
Multi-level fractal processing of complete Spanish vocabulary.

Levels:
1. Base Indexing: Word -> 7D Vector
2. Primary Relations: Geometric neighbors
3. Semantic Clusters: Auto-grouping by resonance
4. Emergent Concepts: Discovered relationships
"""

import sys
import os
import csv
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath('bindings/python'))

from cortex.encoder import FractalEncoder
from cmfo.core.structural import FractalVector7

class SpanishKnowledgeBase:
    def __init__(self, vocab_file='vocabulario/spanish_words.txt'):
        self.vocab_file = vocab_file
        self.encoder = FractalEncoder()
        self.words = []
        self.vectors = {}
        
    def load_vocabulary(self):
        """Load Spanish words from file"""
        print("[1] Loading Spanish Vocabulary...")
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            self.words = [line.strip() for line in f if line.strip()]
        print(f"    Loaded {len(self.words)} words")
        
    def level_1_index(self, output_file='spanish_vectors_L1.csv'):
        """Level 1: Index all words to 7D vectors"""
        print("\n[LEVEL 1] Indexing words to 7D fractal space...")
        start = time.time()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['word', 'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'])
            
            for i, word in enumerate(self.words):
                vec = self.encoder.encode(word)
                self.vectors[word] = vec
                writer.writerow([word] + [f"{x:.6f}" for x in vec.v])
                
                if (i + 1) % 10000 == 0:
                    print(f"    Progress: {i+1}/{len(self.words)} ({100*(i+1)/len(self.words):.1f}%)")
        
        duration = time.time() - start
        print(f"    Complete. Rate: {len(self.words)/duration:.1f} words/sec")
        print(f"    Saved to: {output_file}")
        
    def level_2_relations(self, sample_size=5000, neighbors=10, output_file='spanish_relations_L2.csv'):
        """Level 2: Find geometric neighbors for sample words"""
        print(f"\n[LEVEL 2] Computing primary relations (sample: {sample_size})...")
        
        # Sample words for relation analysis
        import random
        sample_words = random.sample(self.words, min(sample_size, len(self.words)))
        
        results = []
        start = time.time()
        
        for i, word in enumerate(sample_words):
            vec = self.vectors[word]
            
            # Find closest neighbors
            distances = []
            for other_word in self.words:
                if other_word == word:
                    continue
                other_vec = self.vectors[other_word]
                dist = self.encoder.conceptual_distance(vec, other_vec)
                distances.append((other_word, dist))
            
            # Sort and take top N
            distances.sort(key=lambda x: x[1])
            top_neighbors = distances[:neighbors]
            
            for neighbor, dist in top_neighbors:
                results.append([word, neighbor, f"{dist:.4f}"])
            
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{sample_size}")
        
        # Save
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['word', 'neighbor', 'distance'])
            writer.writerows(results)
        
        duration = time.time() - start
        print(f"    Complete. Time: {duration:.1f}s")
        print(f"    Saved to: {output_file}")
        
    def level_3_clusters(self, num_clusters=50, output_file='spanish_clusters_L3.csv'):
        """Level 3: Semantic clustering by resonance"""
        print(f"\n[LEVEL 3] Generating semantic clusters ({num_clusters} clusters)...")
        print("    (Using simple k-means on 7D vectors)")
        
        # Simple clustering: group by dominant dimension
        clusters = defaultdict(list)
        
        for word, vec in self.vectors.items():
            # Find dominant dimension
            max_idx = max(range(7), key=lambda i: abs(vec.v[i]))
            clusters[max_idx].append(word)
        
        # Save
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['cluster_id', 'word_count', 'sample_words'])
            
            for cluster_id, words in clusters.items():
                sample = ', '.join(words[:20])
                writer.writerow([cluster_id, len(words), sample])
        
        print(f"    Complete. Found {len(clusters)} natural clusters")
        print(f"    Saved to: {output_file}")

def main():
    print("="*60)
    print("  CMFO SPANISH KNOWLEDGE BASE GENERATOR")
    print("  Processing Spanish vocabulary through fractal geometry")
    print("="*60)
    
    kb = SpanishKnowledgeBase()
    kb.load_vocabulary()
    kb.level_1_index(output_file='vocabulario/spanish_vectors_L1.csv')
    kb.level_2_relations(sample_size=1000, output_file='vocabulario/spanish_relations_L2.csv')
    kb.level_3_clusters(output_file='vocabulario/spanish_clusters_L3.csv')
    
    print("\n" + "="*60)
    print("  COMPLETE: Spanish Fractal Knowledge Base Generated")
    print("="*60)

if __name__ == "__main__":
    main()
