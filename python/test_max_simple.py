# -*- coding: utf-8 -*-
# Minimal test runner - ASCII only
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from cmfo.universal.constants import PHI
from cmfo.universal.octonion_algebra import (
    Octonion,
    verify_non_associativity,
    verify_alternativity,
)
from cmfo.universal.teleportation_real import TeleportacionRealOctonionica

def main():
    results = []
    
    # Test 1: PHI
    expected_phi = (1 + np.sqrt(5)) / 2
    test1 = abs(PHI - expected_phi) < 1e-15
    results.append(("PHI = Golden Ratio", test1))
    
    # Test 2: Octonion multiplication
    e1 = Octonion.unit(1)
    e2 = Octonion.unit(2)
    producto = e1 * e2
    esperado = np.zeros(8)
    esperado[3] = 1.0
    test2 = np.linalg.norm(producto.c - esperado) < 1e-10
    results.append(("e1*e2 = e3 (Cayley)", test2))
    
    # Test 3: Non-associativity
    na = verify_non_associativity()
    test3 = na['is_non_associative']
    results.append(("O is non-associative", test3))
    
    # Test 4: Alternativity
    alt = verify_alternativity()
    test4 = alt['left_holds'] and alt['right_holds']
    results.append(("O is alternative", test4))
    
    # Test 5: Teleportation with variable fidelity
    tp = TeleportacionRealOctonionica()
    fidelidades = []
    for i in range(5):
        estado = tp.crear_estado_puro(estructura_milnor=i)
        par = tp.crear_par_epr_octonionico(i, (i + 7) % 28)
        resultado = tp.teleportar(estado, par)
        fidelidades.append(resultado['fidelidad'])
    std = np.std(fidelidades)
    test5 = std > 1e-10  # Not constant = not mock
    results.append((f"Teleport fidelity varies (std={std:.4f})", test5))
    
    # Print results - flush immediately
    import sys
    sys.stdout.flush()
    print("\n" + "="*60, flush=True)
    print("MAXIMUM LEVEL VERIFICATION - CMFO ULTRA-7D", flush=True)
    print("="*60, flush=True)
    
    passed = 0
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")
        if result:
            passed += 1
        else:
            print(f"       ^ FAILED")
    
    print("="*60)
    print(f"RESULTS: {passed}/{len(results)} PASSED")
    print("="*60)
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    
    # Also write to file
    with open("test_output.txt", "w") as f:
        import numpy as np
        from cmfo.universal.constants import PHI
        from cmfo.universal.octonion_algebra import Octonion, verify_non_associativity, verify_alternativity
        from cmfo.universal.teleportation_real import TeleportacionRealOctonionica
        
        # Test details
        expected_phi = (1 + np.sqrt(5)) / 2
        f.write(f"PHI test: PHI={PHI}, expected={expected_phi}, diff={abs(PHI - expected_phi)}\n")
        
        e1 = Octonion.unit(1)
        e2 = Octonion.unit(2)
        producto = e1 * e2
        esperado = np.zeros(8)
        esperado[3] = 1.0
        f.write(f"Octonion test: producto={producto.c}, esperado={esperado}, error={np.linalg.norm(producto.c - esperado)}\n")
        
        na = verify_non_associativity()
        f.write(f"Non-assoc test: diff={na['difference_norm']}, is_non_assoc={na['is_non_associative']}\n")
        
        alt = verify_alternativity()
        f.write(f"Alternativity test: left={alt['left_alternative_error']}, right={alt['right_alternative_error']}\n")
        
        tp = TeleportacionRealOctonionica()
        fidelidades = []
        for i in range(5):
            estado = tp.crear_estado_puro(estructura_milnor=i)
            par = tp.crear_par_epr_octonionico(i, (i + 7) % 28)
            resultado = tp.teleportar(estado, par)
            fidelidades.append(resultado['fidelidad'])
        f.write(f"Teleport test: fidelidades={fidelidades}, std={np.std(fidelidades)}\n")
    
    exit(0 if success else 1)
