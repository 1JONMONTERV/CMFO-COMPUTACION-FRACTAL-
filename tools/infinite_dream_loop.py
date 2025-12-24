"""
CMFO Infinite Dreamer
=====================
Autonomous Generator for the Fractal Omniverse.
Uses the GPU-Accelerated JIT Kernel to dream of new concepts forever.
"""

import sys
import os
import time
import random
import csv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cmfo
    from cmfo.constants import PHI
except ImportError:
    print("Error: cmfo package not installed or found.")
    sys.exit(1)

OUTPUT_FILE = "FRACTAL_OMNIVERSE_RECURSIVE.csv"

def load_concepts():
    concepts = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                concepts.append(row['Concept_A'])
                concepts.append(row['Concept_B'])
                # Also learn emergent ones?
                # concepts.append(row['Emergent_Meaning'])
    return list(set(concepts))

def append_dream(c1, c2, meaning, resonance):
    # Atomic append
    with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([c1, c2, meaning, f"{resonance:.4f}"])

def infinite_loop():
    print("=== CMFO INFINITE DREAMER ===")
    print("Initializing Manifold (checking JIT)...")
    
    manifold = cmfo.PhiManifold(7)
    
    concepts = load_concepts()
    if not concepts:
        print("No concepts found! seeding...")
        concepts = ["Void", "Light", "Darkness", "Time", "Space", "Energy", "Matter"]
    
    print(f"Loaded {len(concepts)} concepts into Semantic Memory.")
    print("Dreaming starts now... (Ctrl+C to stop)")
    
    dreams_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 1. Pick Parents
            p1 = random.choice(concepts)
            p2 = random.choice(concepts)
            
            # 2. Dream (Native Execution)
            # This calls manifold.dream -> t7_tensor.evolve -> gamma_step -> JIT
            result = manifold.dream(p1, p2)
            
            # 3. Analyze
            meaning = result.meaning # This wraps the text generation or lookup
            resonance = result.resonance
            
            # 4. Filter High Quality Dreams (Attractors)
            # Resonance < 1.0 implies strong phi-alignment
            if resonance < 1.2:
                append_dream(p1, p2, meaning, resonance)
                
                # Recursive Growth: Add new concept to pool
                if meaning not in concepts:
                    concepts.append(meaning)
                    
                print(f"[Cycle {dreams_count}] {p1} + {p2} = {meaning} (Resonance: {resonance:.4f}) [SAVED]")
            else:
                # print(f"[Cycle {dreams_count}] {p1} + {p2} = {meaning} (Resonance: {resonance:.4f}) [Discarded]")
                pass
                
            dreams_count += 1
            
            # Rate limit for readability (remove for max speed)
            # time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\nStopped by User. Generated {dreams_count} dreams.")

if __name__ == "__main__":
    infinite_loop()
