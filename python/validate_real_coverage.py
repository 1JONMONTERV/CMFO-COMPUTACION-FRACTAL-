#!/usr/bin/env python3
"""
Validación de Cobertura con Nonces Reales

Verifica que TODOS los nonces reales de los 100 bloques están
dentro del espacio reducido por NonceRestrictor.
"""

import os
import sys
import json
import struct
import time

sys.path.insert(0, os.path.dirname(__file__))

from cmfo.bitcoin import NonceRestrictor, build_header

def main():
    print("=" * 70)
    print("  VALIDACIÓN DE COBERTURA - NONCES REALES")
    print("=" * 70)
    
    # 1. Cargar nonces reales
    analysis_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nonce_analysis_real.json')
    
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    nonces = data['nonces']
    print(f"\n[INFO] Nonces reales cargados: {len(nonces)}")
    
    # 2. Validar con diferentes modos
    modes = ['conservative', 'aggressive']
    
    for mode in modes:
        print(f"\n{'='*70}")
        print(f"  MODO: {mode.upper()}")
        print(f"{'='*70}")
        
        # Mock header (los campos no afectan la restricción del nonce)
        header = build_header(
            version=536870912,
            prev_block=bytes(32),
            merkle_root=bytes(32),
            timestamp=int(time.time()),
            bits=0x1715a35c,
            nonce=0
        )
        
        restrictor = NonceRestrictor(header, empirical_mode=mode)
        success, reduced_space, reduction_factor = restrictor.reduce_space()
        
        print(f"\nEspacio original:  2^32 = {2**32:,}")
        print(f"Espacio reducido:  {reduced_space:,}")
        print(f"Factor reducción:  {reduction_factor:.2f}x")
        
        # 3. Verificar cobertura
        print(f"\nVerificando cobertura...")
        
        covered = 0
        not_covered = []
        
        for i, nonce in enumerate(nonces):
            if restrictor.is_nonce_in_space(nonce):
                covered += 1
            else:
                not_covered.append((i, nonce))
        
        coverage_pct = (covered / len(nonces)) * 100
        
        print(f"\nNonces cubiertos:  {covered}/{len(nonces)} ({coverage_pct:.1f}%)")
        
        if not_covered:
            print(f"\n⚠️  NONCES NO CUBIERTOS:")
            for idx, nonce in not_covered[:10]:  # Mostrar primeros 10
                nonce_bytes = struct.pack('<I', nonce)
                print(f"  Bloque #{idx}: 0x{nonce:08x} = [{nonce_bytes[0]:3d}, {nonce_bytes[1]:3d}, {nonce_bytes[2]:3d}, {nonce_bytes[3]:3d}]")
            if len(not_covered) > 10:
                print(f"  ... y {len(not_covered) - 10} más")
        else:
            print(f"\n✅ COBERTURA 100% - Todos los nonces reales están cubiertos!")
        
        # 4. Mostrar rangos aplicados
        print(f"\nRangos aplicados:")
        domains = restrictor.get_nonce_domains()
        for pos in sorted(domains.keys()):
            domain = domains[pos]
            byte_idx = pos - 76
            print(f"  Byte {byte_idx}: {len(domain)} valores")
    
    print(f"\n{'='*70}")
    print("  VALIDACIÓN COMPLETADA")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
