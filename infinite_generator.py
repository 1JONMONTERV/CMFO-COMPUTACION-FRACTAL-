
import csv
import time
import sys
import math
import random
from datetime import datetime
import os

CSV_PATH = "FRACTAL_OMNIVERSE_RECURSIVE.csv"

# Pre-defined meanings to keep semantic coherence high
MEANINGS = [
    "Poder", "Verdad", "Silencio", "Caos", "Orden", "Belleza", "Vacío", 
    "Tiempo", "Espacio", "Energía", "Vida", "Muerte", "Fractal", "Geometría",
    "Lógica", "Mente", "Alma", "Entropía", "Gravedad", "Luz", "Oscuridad",
    "Resonancia", "Armonía", "Conflicto", "Evolución", "Singularidad",
    "Ilusión", "Realidad", "Sueño", "Despertar", "Transformación"
]

def get_existing_concepts():
    concepts = set()
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # header
            for row in reader:
                if len(row) >= 2:
                    concepts.add(row[0])
                    concepts.add(row[1])
    except FileNotFoundError:
        pass
    
    if not concepts:
        return ["Time", "Space", "Matter", "Energy", "Void"] # Seed
    return list(concepts)

def infinite_loop(max_gens=None):
            print(f"[*] Iniciando Generador Semántico Infinito en {CSV_PATH}")
    
    concepts = get_existing_concepts()
    print(f"[*] Se cargaron {len(concepts)} conceptos únicos.")
    
    gen_count = 0
    while True:
        try:
            # 1. Select Parents
            c1 = random.choice(concepts)
            c2 = random.choice(concepts)
            if c1 == c2: continue
            
            # 2. Determine Meaning
            meaning = random.choice(MEANINGS)
            
            # 3. Calculate Resonance (Fractal Pseudo-Random)
            # Resonance often around Phi (1.618) or its powers
            phi = 1.618034
            base = random.choice([phi, phi**2, phi**-1, 1.0, math.pi])
            noise = random.uniform(-0.5, 0.5)
            resonance = abs(base + noise)
            
            # 4. Append
            row = [c1, c2, meaning, f"{resonance:.4f}"]
            
            with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            gen_count += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {c1} + {c2} -> {meaning} ({resonance:.2f})", end='\r')
            
            # 5. Possibly add new concept (Evolution)
            if random.random() < 0.05: # 5% chance of mutation
                new_concept = f"{meaning}_{random.randint(1,99)}"
                concepts.append(new_concept)
                
            if max_gens and gen_count >= max_gens:
                break
                
            time.sleep(0.5) # 2Hz
            
        except KeyboardInterrupt:
            print("\n[!] Stopped by user.")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    infinite_loop()
