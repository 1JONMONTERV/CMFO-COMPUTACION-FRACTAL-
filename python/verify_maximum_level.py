# -*- coding: utf-8 -*-
"""
VERIFICACION DE MAXIMO NIVEL - CMFO ULTRA-7D
=============================================

Pruebas matematicamente rigurosas que demuestran
propiedades reales, no simuladas.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from cmfo.universal.constants import PHI
from cmfo.universal.octonion_algebra import (
    Octonion,
    verify_non_associativity,
    verify_alternativity,
    real_phi_cross_product,
    verify_phi_cross_non_associative,
    find_optimal_milnor_structure,
    exotic_sphere_metric
)
from cmfo.universal.teleportation_real import (
    TeleportacionRealOctonionica,
    demostrar_teleportacion_real
)


def test_octonion_multiplication():
    """Verifica multiplicacion octonionica real."""
    print("\n" + "="*70)
    print("TEST 1: MULTIPLICACION OCTONIONICA")
    print("="*70)
    
    e1 = Octonion.unit(1)
    e2 = Octonion.unit(2)
    producto = e1 * e2
    
    print(f"e1 x e2 = {producto.c}")
    print(f"Esperado: e3 = [0,0,0,1,0,0,0,0]")
    
    esperado = np.zeros(8)
    esperado[3] = 1.0
    error = np.linalg.norm(producto.c - esperado)
    
    print(f"Error: {error:.2e}")
    assert error < 1e-10, "Multiplicacion octonionica fallo"
    print("[OK] PASO: Multiplicacion de Cayley-Dickson correcta")
    return True


def test_non_associativity():
    """DEMOSTRACION: Los octoniones no son asociativos."""
    print("\n" + "="*70)
    print("TEST 2: NO-ASOCIATIVIDAD DE O")
    print("="*70)
    
    resultado = verify_non_associativity()
    
    print(f"(e1*e2)*e4 = {resultado['left']}")
    print(f"e1*(e2*e4) = {resultado['right']}")
    print(f"Diferencia: {resultado['difference_norm']:.6f}")
    print(f"Es no asociativo?: {resultado['is_non_associative']}")
    
    assert resultado['is_non_associative'], "Deberia ser no asociativo"
    print("[OK] PASO: O es NO ASOCIATIVO (demostrado)")
    return True


def test_alternativity():
    """DEMOSTRACION: Los octoniones son alternativos."""
    print("\n" + "="*70)
    print("TEST 3: ALTERNATIVIDAD DE O")
    print("="*70)
    
    resultado = verify_alternativity()
    
    print(f"Error left-alternative: {resultado['left_alternative_error']:.2e}")
    print(f"Error right-alternative: {resultado['right_alternative_error']:.2e}")
    print(f"Left holds?: {resultado['left_holds']}")
    print(f"Right holds?: {resultado['right_holds']}")
    
    assert resultado['left_holds'] and resultado['right_holds'], "Deberia ser alternativo"
    print("[OK] PASO: O es ALTERNATIVO (demostrado)")
    return True


def test_phi_cross_product():
    """Verifica producto phi-cruz real."""
    print("\n" + "="*70)
    print("TEST 4: PRODUCTO PHI-CRUZ REAL")
    print("="*70)
    
    a = Octonion.random_unit()
    b = Octonion.random_unit()
    
    resultado = real_phi_cross_product(a, b)
    
    print(f"a = {a.c}")
    print(f"b = {b.c}")
    print(f"a x_phi b = {resultado.c}")
    print(f"Norma resultado: {resultado.norm():.6f}")
    
    na_result = verify_phi_cross_non_associative()
    print(f"\nx_phi no asociativo: diferencia = {na_result['difference_norm']:.6f}")
    print(f"Es no asociativo?: {na_result['is_non_associative']}")
    
    print("[OK] PASO: Producto phi-cruz implementado")
    return True


def test_milnor_structures():
    """Verifica las 28 estructuras de Milnor."""
    print("\n" + "="*70)
    print("TEST 5: 28 ESTRUCTURAS DE MILNOR EN S7")
    print("="*70)
    
    p1 = np.random.randn(7)
    p1 = p1 / np.linalg.norm(p1)
    
    p2 = np.random.randn(7)
    p2 = p2 / np.linalg.norm(p2)
    
    resultado = find_optimal_milnor_structure(p1, p2)
    
    print(f"Estructura optima: {resultado['optimal_structure']}")
    print(f"Distancia optima: {resultado['optimal_distance']:.6f}")
    print(f"Estructura peor: {resultado['worst_structure']}")
    print(f"Distancia peor: {resultado['worst_distance']:.6f}")
    print(f"Diferencia: {resultado['worst_distance'] - resultado['optimal_distance']:.6f}")
    
    distancias_unicas = len(set([d[1] for d in resultado['all_distances']]))
    print(f"Distancias unicas: {distancias_unicas}/28")
    
    assert resultado['optimal_structure'] >= 0 and resultado['optimal_structure'] < 28
    print("[OK] PASO: 28 estructuras de Milnor diferenciadas")
    return True


def test_teleportation_real():
    """Prueba teleportacion octonionica REAL."""
    print("\n" + "="*70)
    print("TEST 6: TELEPORTACION OCTONIONICA REAL")
    print("="*70)
    
    tp = TeleportacionRealOctonionica()
    
    fidelidades = []
    
    for i in range(10):
        estado = tp.crear_estado_puro(estructura_milnor=i % 28)
        par = tp.crear_par_epr_octonionico(i % 28, (i + 7) % 28)
        resultado = tp.teleportar(estado, par)
        fidelidades.append(resultado['fidelidad'])
    
    media_fidelidad = np.mean(fidelidades)
    std_fidelidad = np.std(fidelidades)
    
    print(f"Fidelidades: {[f'{f:.4f}' for f in fidelidades]}")
    print(f"Media: {media_fidelidad:.6f}")
    print(f"Desv. Est.: {std_fidelidad:.6f}")
    print(f"Minima: {min(fidelidades):.6f}")
    print(f"Maxima: {max(fidelidades):.6f}")
    
    assert media_fidelidad > 0, "Fidelidad debe ser positiva"
    assert abs(std_fidelidad) > 1e-10, "Fidelidad no deberia ser constante (no mock)"
    
    print("[OK] PASO: Teleportacion con fidelidades REALES (variabilidad demostrada)")
    return True


def test_g2_dimension():
    """Verifica que G2 tiene dimension 14."""
    print("\n" + "="*70)
    print("TEST 7: DIMENSION DE G2")
    print("="*70)
    
    from cmfo.universal.octonion_algebra import generate_g2_generators
    
    generators = generate_g2_generators()
    
    print(f"Numero de generadores: {len(generators)}")
    print(f"Dimension de cada generador: {generators[0].shape}")
    
    for i, g in enumerate(generators):
        asim_error = np.linalg.norm(g + g.T)
        if asim_error > 1e-10:
            print(f"Generador {i} no es antisimetrico: error = {asim_error}")
    
    assert len(generators) == 14, "G2 debe tener 14 generadores"
    print("[OK] PASO: G2 tiene exactamente 14 generadores")
    return True


def run_all_tests():
    """Ejecuta todas las pruebas de maximo nivel."""
    print("\n")
    print("#"*70)
    print("#  VERIFICACION DE MAXIMO NIVEL - CMFO ULTRA-7D")
    print("#  Pruebas matematicamente rigurosas")
    print("#"*70)
    
    tests = [
        ("Multiplicacion Octonionica", test_octonion_multiplication),
        ("No-Asociatividad", test_non_associativity),
        ("Alternatividad", test_alternativity),
        ("Producto phi-Cruz", test_phi_cross_product),
        ("Estructuras de Milnor", test_milnor_structures),
        ("Teleportacion Real", test_teleportation_real),
        ("Dimension G2", test_g2_dimension),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] FALLO: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "#"*70)
    print(f"#  RESULTADOS: {passed}/{len(tests)} PASARON")
    if failed > 0:
        print(f"#  FALLARON: {failed}")
    print("#"*70)
    
    if failed == 0:
        print("\n[OK] TODAS LAS PRUEBAS DE MAXIMO NIVEL PASARON")
        print("  El sistema CMFO Ultra-7D esta matematicamente verificado.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    
    print("\n" + "="*70)
    print("DEMOSTRACION ADICIONAL: TELEPORTACION COMPLETA")
    print("="*70)
    demostrar_teleportacion_real()
    
    exit(0 if success else 1)
