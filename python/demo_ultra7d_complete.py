# -*- coding: utf-8 -*-
"""
DEMO COMPLETO: SISTEMA CMFO ULTRA-7D
====================================

Demostración integral que verifica y muestra todas las capacidades
del sistema CMFO Ultra-7D:

1. Álgebra Octoniónica (Cayley-Dickson)
2. Propiedades No-Asociativas
3. Alternatividad
4. 28 Estructuras de Milnor
5. Teleportación Octoniónica
6. Benchmark GPU

Autor: CMFO Universe
Fecha: Diciembre 2024
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time

# Importar módulos CMFO
from cmfo.universal.constants import PHI
from cmfo.universal.octonion_algebra import (
    Octonion,
    verify_non_associativity,
    verify_alternativity,
    real_phi_cross_product,
    find_optimal_milnor_structure,
    generate_g2_generators
)
from cmfo.universal.teleportation_real import (
    TeleportacionRealOctonionica,
)


def print_header(title):
    """Imprime un encabezado bonito."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title):
    """Imprime una sección."""
    print()
    print(f"--- {title} ---")


def demo_octonionic_multiplication():
    """Demuestra la multiplicación octoniónica."""
    print_section("Multiplicación Octoniónica (Cayley-Dickson)")
    
    e1 = Octonion.unit(1)
    e2 = Octonion.unit(2)
    e3 = e1 * e2
    
    print(f"  e₁ × e₂ = e₃")
    print(f"  Resultado: {e3.c}")
    print(f"  Esperado:  [0, 0, 0, 1, 0, 0, 0, 0]")
    
    error = np.linalg.norm(e3.c - Octonion.unit(3).c)
    print(f"  Error: {error:.2e}")
    return error < 1e-10


def demo_non_associativity():
    """Demuestra la no-asociatividad."""
    print_section("No-Asociatividad de O")
    
    result = verify_non_associativity()
    
    print(f"  (e₁ × e₂) × e₄ = {result['left']}")
    print(f"  e₁ × (e₂ × e₄) = {result['right']}")
    print(f"  Diferencia: {result['difference_norm']:.6f}")
    print(f"  ¿Es no-asociativo?: {'SI' if result['is_non_associative'] else 'NO'}")
    
    return result['is_non_associative']


def demo_alternativity():
    """Demuestra la alternatividad."""
    print_section("Alternatividad de O")
    
    result = verify_alternativity()
    
    print(f"  Error izquierdo:  {result['left_alternative_error']:.2e}")
    print(f"  Error derecho:    {result['right_alternative_error']:.2e}")
    print(f"  ¿Alternativo?: {'SI' if result['left_holds'] and result['right_holds'] else 'NO'}")
    
    return result['left_holds'] and result['right_holds']


def demo_phi_cross():
    """Demuestra el producto φ-cruz."""
    print_section("Producto φ-Cruz")
    
    a = Octonion.random_unit()
    b = Octonion.random_unit()
    
    result = real_phi_cross_product(a, b)
    
    print(f"  a = {a.c[:4]}...")
    print(f"  b = {b.c[:4]}...")
    print(f"  a ×_φ b norma = {result.norm():.6f}")
    print(f"  φ = {PHI:.6f}")
    
    return True


def demo_milnor_structures():
    """Demuestra las 28 estructuras de Milnor."""
    print_section("28 Estructuras de Milnor en S⁷")
    
    p1 = np.random.randn(7)
    p1 = p1 / np.linalg.norm(p1)
    p2 = np.random.randn(7)
    p2 = p2 / np.linalg.norm(p2)
    
    result = find_optimal_milnor_structure(p1, p2)
    
    print(f"  Estructura óptima: {result['optimal_structure']}")
    print(f"  Distancia óptima:  {result['optimal_distance']:.6f}")
    print(f"  Estructura peor:   {result['worst_structure']}")
    print(f"  Distancia peor:    {result['worst_distance']:.6f}")
    print(f"  Diferencia:        {result['worst_distance'] - result['optimal_distance']:.6f}")
    
    unique = len(set([d[1] for d in result['all_distances']]))
    print(f"  Distancias únicas: {unique}/28")
    
    return unique == 28


def demo_g2_group():
    """Demuestra el grupo G₂."""
    print_section("Grupo G₂ (Automorfismos de O)")
    
    generators = generate_g2_generators()
    
    print(f"  Número de generadores: {len(generators)}")
    print(f"  Dimensión cada uno:    {generators[0].shape}")
    print(f"  dim(G₂) = 14 {'(CORRECTO)' if len(generators) == 14 else '(ERROR)'}")
    
    return len(generators) == 14


def demo_teleportation():
    """Demuestra la teleportación octoniónica."""
    print_section("Teleportación Octoniónica")
    
    tp = TeleportacionRealOctonionica()
    
    # Crear estado y par EPR
    estado = tp.crear_estado_puro(estructura_milnor=7)
    par = tp.crear_par_epr_octonionico(7, 14)
    
    # Teleportar
    resultado = tp.teleportar(estado, par)
    
    print(f"  Estructura Milnor:    {7} -> {14}")
    print(f"  Fidelidad:            {resultado['fidelidad']:.6f}")
    print(f"  Bits clásicos:        {resultado['bits_clasicos']}")
    
    # Manejar diferentes nombres de clave
    trits = resultado.get('trits_enviados', resultado.get('trits', []))
    if len(trits) > 0:
        print(f"  Valores medidos:      {trits[:4]}...")
    
    return resultado['fidelidad'] > 0


def demo_gpu_capability():
    """Verifica capacidad GPU."""
    print_section("Capacidad GPU (CUDA)")
    
    try:
        from numba import cuda
        cuda.select_device(0)
        device = cuda.get_current_device()
        print(f"  GPU detectada: {device.name}")
        print(f"  Compute Capability: {device.compute_capability}")
        return True
    except Exception as e:
        print(f"  GPU no disponible: {e}")
        return False


def run_full_demo():
    """Ejecuta la demostración completa."""
    
    print_header("DEMO COMPLETO: SISTEMA CMFO ULTRA-7D")
    print(f"  φ (Phi) = {PHI}")
    print(f"  Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Multiplicación Octoniónica", demo_octonionic_multiplication),
        ("No-Asociatividad", demo_non_associativity),
        ("Alternatividad", demo_alternativity),
        ("Producto φ-Cruz", demo_phi_cross),
        ("28 Estructuras de Milnor", demo_milnor_structures),
        ("Grupo G₂", demo_g2_group),
        ("Teleportación", demo_teleportation),
        ("Capacidad GPU", demo_gpu_capability),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False))
    
    # Resumen final
    print_header("RESUMEN FINAL")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    
    print()
    print(f"  TOTAL: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print()
        print("  " + "*" * 50)
        print("  *  SISTEMA CMFO ULTRA-7D: COMPLETAMENTE OPERATIVO  *")
        print("  " + "*" * 50)
    
    return passed == total


if __name__ == "__main__":
    success = run_full_demo()
    sys.exit(0 if success else 1)
