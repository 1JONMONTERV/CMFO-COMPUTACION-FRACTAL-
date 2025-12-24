# -*- coding: utf-8 -*-
"""
DEMO: BASE DE CONOCIMIENTO FRACTAL
==================================

Demostración de la Tabla de Conocimiento Procedural Infinita.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cmfo.universal.knowledge_table import ProceduralKnowledgeTable

def run_knowledge_demo():
    print("=== INICIANDO SISTEMA DE CONOCIMIENTO TOTAL ===")
    
    kt = ProceduralKnowledgeTable()
    
    # 1. Ingesta de Conocimiento (Learning)
    conceptos = [
        ("Octonión", "Algebra de division normada de dimension 8, no asociativa."),
        ("S7", "Esfera de dimension 7, paralelizable y con estructuras exóticas."),
        ("Milnor", "Topologo que descubrio las esferas exoticas en S7."),
        ("Phi", "Proporcion aurea, base de la resonancia fractal."),
        ("Fano", "Plano proyectivo finito de orden 2, define multiplicacion."),
        ("G2", "Grupo de Lie simple mas pequeno, automorfismos de octoniones."),
        ("Teleportación", "Transferencia de estado cuantico usando entrelazamiento.")
    ]
    
    print(f"\n[LEARN] Procesando {len(conceptos)} conceptos...")
    
    for term, definition in conceptos:
        meta = kt.learn(term, definition)
        print(f"  + '{term}' almacenado en Capa Milnor {meta['milnor_layer']}")
        # print(f"    Hash: {meta['raw_hash'][:16]}...")
    
    # 2. Recuperación (Recall)
    print(f"\n[RECALL] Verificando integridad...")
    term = "Octonión"
    definicion = kt.recall(term)
    print(f"  Buscando '{term}':")
    print(f"  -> {definicion}")
    
    if definicion == conceptos[0][1]:
        print("  [OK] Recuperación exacta.")
    else:
        print("  [FAIL] Error en recuperación.")
    
    # 3. Relaciones Geométricas (Semantic Distance)
    print(f"\n[RELATIONS] Buscando conceptos cercanos en S7...")
    target = "Octonión"
    relacionados = kt.find_related(target, n=3)
    
    print(f"  Conceptos cercanos a '{target}':")
    for term, dist in relacionados:
        print(f"  - {term}: dist = {dist:.4f} rad")
    
    # 4. Hashing Determinista
    print(f"\n[CONSISTENCY] Verificando determinismo del hash...")
    h1 = kt.hasher.hash_concept("Caos")
    h2 = kt.hasher.hash_concept("Caos")
    
    dist_error = np.linalg.norm(h1['coords'] - h2['coords'])
    if dist_error < 1e-15:
        print(f"  [OK] Hash determinista (error={dist_error:.2e})")
    else:
        print(f"  [FAIL] Hash no determinista!")

    # 5. Capacidad
    mem_bits = sum(len(c.trajectory) for c in kt.memory.channels)
    print(f"\n[STATS]")
    print(f"  Total operaciones octoniónicas almacenadas: {mem_bits}")
    print(f"  Capas Milnor utilizadas: {len(set(idx['layer'] for idx in kt.index.values()))}")


if __name__ == "__main__":
    import numpy as np # Needed for the check
    run_knowledge_demo()
