
import csv
import json
import os
import sys

class FractalCompressor:
    """
    Holographic Compressor.
    Reduces the 'Fractal Omniverse' to its absolute minimum entropy state (The Seed).
    """
    def compress(self, csv_path, output_path):
        print(f"[*] Compressing Holographic Knowledge from {csv_path}...")
        
        unique_concepts = set()
        relation_count = 0
        
        # 1. Harvest Axioms (Unique Concepts)
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    unique_concepts.add(row['Concept_A'])
                    unique_concepts.add(row['Concept_B'])
                    # Note: We don't even need to store the 'Emergent_Meaning' 
                    # because the engine can deterministicall RE-DREAM it on demand.
                    # That is the power of Deterministic AI.
                    relation_count += 1
        except FileNotFoundError:
            print("Error: CSV not found.")
            return

        sorted_axioms = sorted(list(unique_concepts))
        
        # 2. Statistics
        original_size = os.path.getsize(csv_path)
        
        # 3. Create Seed Package
        seed_package = {
            "meta": {
                "engine_version": "v4.0",
                "axiom_count": len(sorted_axioms),
                "virtual_relations": relation_count
            },
            "axioms": sorted_axioms
        }
        
        # 4. Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(seed_package, f, indent=2)
            
        compressed_size = os.path.getsize(output_path)
        ratio = original_size / compressed_size
        
        print(f"[*] Compressed {relation_count} relations to {len(sorted_axioms)} axiomas.")
        print(f"    Original Size:   {original_size} bytes")
        print(f"    Compressed Size: {compressed_size} bytes")
        print(f"    Compression Ratio: {ratio:.2f}x")
        print(f"    Portable Seed saved to: {output_path}")

if __name__ == "__main__":
    comp = FractalCompressor()
    comp.compress("FRACTAL_OMNIVERSE_RECURSIVE.csv", "FRACTAL_SEED.json")
