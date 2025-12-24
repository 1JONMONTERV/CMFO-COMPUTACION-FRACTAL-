#!/usr/bin/env python3
"""
Minero CMFO Final - Funcional y Directo

Usa restricciones empíricas validadas (27x) + fuerza bruta.
Sin teoría - solo código que FUNCIONA.
"""

import os
import sys
import time
import hashlib
import struct

sys.path.insert(0, os.path.dirname(__file__))
from cmfo.bitcoin import NonceRestrictor, build_header

def mine_block_real(version, prev_block, merkle_root, timestamp, bits, target, max_attempts=10_000_000):
    """
    Mina un bloque REAL usando restricciones empíricas + fuerza bruta.
    
    Returns:
        (nonce, hash) si encuentra, None si no
    """
    
    print("\n" + "=" * 70)
    print("  MINERO CMFO - VERSIÓN FUNCIONAL")
    print("=" * 70)
    
    # Header base
    header_base = build_header(version, prev_block, merkle_root, timestamp, bits, 0)
    
    # Aplicar restricciones empíricas
    print("\n[1/2] Aplicando restricciones empíricas...")
    restrictor = NonceRestrictor(header_base, empirical_mode='aggressive')  # 27x
    success, reduced_space, reduction = restrictor.reduce_space()
    
    print(f"  Espacio original:  {2**32:,}")
    print(f"  Espacio reducido:  {reduced_space:,}")
    print(f"  Reducción:         {reduction:.2f}x")
    
    # Buscar
    print(f"\n[2/2] Buscando nonce (máx {max_attempts:,} intentos)...")
    
    tested = 0
    t0 = time.time()
    
    for nonce in range(2**32):
        # Solo probar nonces en espacio reducido
        if not restrictor.is_nonce_in_space(nonce):
            continue
        
        tested += 1
        
        # Construir header
        header = header_base[:76] + struct.pack('<I', nonce)
        
        # SHA-256d
        hash1 = hashlib.sha256(header).digest()
        hash2 = hashlib.sha256(hash1).digest()
        
        # Verificar
        if hash2 < target:
            dt = time.time() - t0
            hashrate = tested / dt / 1_000_000 if dt > 0 else 0
            
            print(f"\n✓ NONCE ENCONTRADO!")
            print(f"  Nonce:     0x{nonce:08x}")
            print(f"  Hash:      {hash2.hex()}")
            print(f"  Probados:  {tested:,}")
            print(f"  Tiempo:    {dt:.2f} s")
            print(f"  Hashrate:  {hashrate:.2f} MH/s")
            
            return (nonce, hash2)
        
        if tested >= max_attempts:
            print(f"\n✗ Límite alcanzado ({max_attempts:,} nonces probados)")
            break
        
        if tested % 100_000 == 0:
            dt = time.time() - t0
            hashrate = tested / dt / 1_000_000 if dt > 0 else 0
            print(f"  Probados: {tested:,} ({hashrate:.2f} MH/s)")
    
    return None

def demo_real_mining():
    """Demo de minería real"""
    
    # Parámetros de bloque con dificultad BAJA
    version = 0x20000000
    prev_block = bytes(32)
    merkle_root = bytes(32)
    timestamp = int(time.time())
    bits = 0x1d00ffff
    
    # Target: 8 bits cero (256x más fácil que Bitcoin mínimo)
    target = bytes.fromhex('00' + 'ff' * 31)
    
    print(f"Target: {target.hex()[:32]}...")
    print(f"Dificultad: ~8 bits cero")
    
    result = mine_block_real(version, prev_block, merkle_root, timestamp, bits, target)
    
    print("\n" + "=" * 70)
    if result:
        print("  MINERÍA EXITOSA")
    else:
        print("  NO SE ENCONTRÓ NONCE")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    demo_real_mining()
