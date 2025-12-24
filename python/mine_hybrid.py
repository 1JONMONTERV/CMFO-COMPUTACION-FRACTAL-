#!/usr/bin/env python3
"""
Minero Híbrido CMFO

Integra:
1. Análisis estructural profundo de SHA-256
2. Restricciones empíricas de bloques reales  
3. Búsqueda GPU en espacio reducido

Pipeline:
CPU (Análisis Estructural) → CPU (Reducción Empírica) → GPU (Fuerza Bruta)
"""

import os
import sys
import time
import struct
import hashlib
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from cmfo.bitcoin import NonceRestrictor, build_header
from cmfo.bitcoin.sha256_structural import SHA256StructuralAnalyzer
from cmfo.bitcoin.sha256_inverse import SHA256RoundInverter

class HybridMiner:
    """
    Minero híbrido que combina análisis matemático con fuerza bruta GPU.
    """
    
    def __init__(self, use_structural_analysis: bool = True):
        """
        Args:
            use_structural_analysis: Si usar análisis estructural SHA-256
        """
        self.use_structural = use_structural_analysis
        self.stats = {
            'structural_reduction': 1.0,
            'empirical_reduction': 1.0,
            'total_reduction': 1.0,
            'time_structural_ms': 0,
            'time_empirical_ms': 0,
            'time_search_ms': 0
        }
    
    def mine_block(self, 
                   version: int,
                   prev_block: bytes,
                   merkle_root: bytes,
                   timestamp: int,
                   bits: int,
                   target: bytes,
                   max_nonce: int = 2**32) -> Optional[Tuple[int, bytes]]:
        """
        Mina un bloque usando análisis híbrido.
        
        Args:
            version, prev_block, merkle_root, timestamp, bits: Campos del header
            target: Hash objetivo
            max_nonce: Máximo nonce a probar
            
        Returns:
            (nonce, hash) si se encuentra, None si no
        """
        
        print("\n" + "=" * 70)
        print("  MINERO HÍBRIDO CMFO")
        print("=" * 70)
        
        # Construir header base (con nonce=0)
        header_base = build_header(version, prev_block, merkle_root, timestamp, bits, 0)
        
        # FASE 1: Análisis Estructural (si está habilitado)
        structural_constraints = None
        if self.use_structural:
            print("\n[FASE 1] Análisis Estructural SHA-256...")
            t0 = time.time()
            
            # Calcular difficulty bits del target
            difficulty_bits = self._count_leading_zero_bits(target)
            
            # Analizar estructura
            analyzer = SHA256StructuralAnalyzer(target, difficulty_bits)
            reduced_space, details = analyzer.estimate_nonce_space_reduction(
                header_base, 
                analyze_rounds=10
            )
            
            self.stats['structural_reduction'] = details['factor_reduccion']
            self.stats['time_structural_ms'] = (time.time() - t0) * 1000
            
            print(f"  Difficulty: {difficulty_bits} bits cero")
            print(f"  Reducción estructural: {self.stats['structural_reduction']:.2f}x")
            print(f"  Tiempo: {self.stats['time_structural_ms']:.2f} ms")
            
            structural_constraints = details.get('nonce_constraints')
        
        # FASE 2: Restricciones Empíricas
        print("\n[FASE 2] Aplicando Restricciones Empíricas...")
        t0 = time.time()
        
        restrictor = NonceRestrictor(header_base, empirical_mode='conservative')
        success, reduced_space, reduction_factor = restrictor.reduce_space()
        
        self.stats['empirical_reduction'] = reduction_factor
        self.stats['time_empirical_ms'] = (time.time() - t0) * 1000
        
        print(f"  Reducción empírica: {reduction_factor:.2f}x")
        print(f"  Espacio reducido: {reduced_space:,} nonces")
        print(f"  Tiempo: {self.stats['time_empirical_ms']:.2f} ms")
        
        # FASE 3: Búsqueda en Espacio Reducido
        print("\n[FASE 3] Búsqueda en Espacio Reducido...")
        
        # Calcular reducción total
        self.stats['total_reduction'] = (
            self.stats['structural_reduction'] * 
            self.stats['empirical_reduction']
        )
        
        print(f"  Reducción TOTAL: {self.stats['total_reduction']:.2f}x")
        print(f"  Espacio a buscar: {2**32 / self.stats['total_reduction']:,.0f} nonces")
        
        # Buscar (CPU por ahora, GPU después)
        t0 = time.time()
        result = self._search_reduced_space(
            header_base, 
            target, 
            restrictor,
            max_nonce
        )
        self.stats['time_search_ms'] = (time.time() - t0) * 1000
        
        if result:
            nonce, hash_found = result
            print(f"\n✓ NONCE ENCONTRADO: 0x{nonce:08x}")
            print(f"  Hash: {hash_found.hex()}")
            print(f"  Tiempo búsqueda: {self.stats['time_search_ms']:.2f} ms")
        else:
            print(f"\n✗ No se encontró nonce válido")
            print(f"  Tiempo búsqueda: {self.stats['time_search_ms']:.2f} ms")
        
        # Mostrar estadísticas finales
        self._print_stats()
        
        return result
    
    def _count_leading_zero_bits(self, target: bytes) -> int:
        """Cuenta bits cero al inicio del target"""
        count = 0
        for byte in target:
            if byte == 0:
                count += 8
            else:
                # Contar bits cero en este byte
                for i in range(7, -1, -1):
                    if (byte >> i) & 1 == 0:
                        count += 1
                    else:
                        return count
        return count
    
    def _search_reduced_space(self,
                             header_base: bytes,
                             target: bytes,
                             restrictor: NonceRestrictor,
                             max_nonce: int) -> Optional[Tuple[int, bytes]]:
        """
        Busca en el espacio reducido.
        Por ahora usa CPU, después se migrará a GPU.
        """
        
        tested = 0
        found = 0
        
        # Obtener dominios del nonce
        domains = restrictor.get_nonce_domains()
        
        # Generar nonces válidos y probarlos
        for nonce in range(max_nonce):
            # Verificar si el nonce está en el espacio reducido
            if not restrictor.is_nonce_in_space(nonce):
                continue
            
            tested += 1
            
            # Construir header con este nonce
            header = header_base[:76] + struct.pack('<I', nonce)
            
            # Calcular SHA-256d
            hash1 = hashlib.sha256(header).digest()
            hash2 = hashlib.sha256(hash1).digest()
            
            # Verificar si cumple el target
            if hash2 < target:
                return (nonce, hash2)
            
            # Mostrar progreso cada 100k nonces
            if tested % 100000 == 0:
                print(f"  Probados: {tested:,} nonces...")
        
        return None
    
    def _print_stats(self):
        """Imprime estadísticas finales"""
        print("\n" + "=" * 70)
        print("  ESTADÍSTICAS FINALES")
        print("=" * 70)
        print(f"Reducción estructural:  {self.stats['structural_reduction']:>10.2f}x")
        print(f"Reducción empírica:     {self.stats['empirical_reduction']:>10.2f}x")
        print(f"Reducción TOTAL:        {self.stats['total_reduction']:>10.2f}x")
        print("-" * 70)
        print(f"Tiempo estructural:     {self.stats['time_structural_ms']:>10.2f} ms")
        print(f"Tiempo empírico:        {self.stats['time_empirical_ms']:>10.2f} ms")
        print(f"Tiempo búsqueda:        {self.stats['time_search_ms']:>10.2f} ms")
        total_time = (
            self.stats['time_structural_ms'] + 
            self.stats['time_empirical_ms'] + 
            self.stats['time_search_ms']
        )
        print(f"Tiempo TOTAL:           {total_time:>10.2f} ms")
        print("=" * 70)

def demo_hybrid_mining():
    """Demostración del minero híbrido"""
    
    print("\n" + "=" * 70)
    print("  DEMOSTRACIÓN MINERO HÍBRIDO")
    print("=" * 70)
    
    # Crear un bloque de prueba con dificultad BAJA
    # (para que sea factible encontrar un nonce en tiempo razonable)
    
    version = 0x20000000
    prev_block = bytes.fromhex('0' * 64)
    merkle_root = bytes.fromhex('0' * 64)
    timestamp = int(time.time())
    
    # Dificultad muy baja: solo 8 bits cero (256x más fácil que Bitcoin mínimo)
    # Target: 0x00FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    target = bytes.fromhex('00' + 'ff' * 31)
    bits = 0x1d00ffff  # Representación compacta
    
    print(f"\nParámetros del bloque:")
    print(f"  Version: 0x{version:08x}")
    print(f"  Timestamp: {timestamp}")
    print(f"  Target: {target.hex()[:32]}...")
    print(f"  Dificultad: ~8 bits cero")
    
    # Minar
    miner = HybridMiner(use_structural_analysis=True)
    result = miner.mine_block(
        version, prev_block, merkle_root, timestamp, bits, target,
        max_nonce=10_000_000  # Limitar búsqueda a 10M nonces
    )
    
    if result:
        nonce, hash_found = result
        print(f"\n{'='*70}")
        print("  ✓ MINERÍA EXITOSA")
        print(f"{'='*70}")
        print(f"Nonce: 0x{nonce:08x} ({nonce})")
        print(f"Hash:  {hash_found.hex()}")
    else:
        print(f"\n{'='*70}")
        print("  ✗ No se encontró nonce en el rango especificado")
        print(f"{'='*70}")

if __name__ == "__main__":
    demo_hybrid_mining()
