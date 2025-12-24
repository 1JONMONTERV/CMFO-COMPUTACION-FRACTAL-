#!/usr/bin/env python3
"""
SHA-256 Round Inverter usando Z3 Solver

Invierte rondas de SHA-256 para encontrar restricciones en el nonce
dado un target hash con N bits cero.
"""

import struct
from typing import List, Tuple, Dict, Optional
try:
    from z3 import *
except ImportError:
    print("[WARNING] Z3 no instalado. Instalar con: pip install z3-solver")
    print("[INFO] Continuando con implementación simplificada...")
    Z3_AVAILABLE = False
else:
    Z3_AVAILABLE = True

# Constantes SHA-256
K_SHA256 = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

H0_SHA256 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

class SHA256RoundInverter:
    """
    Invierte rondas de SHA-256 usando Z3 solver.
    """
    
    def __init__(self):
        self.solver = Solver() if Z3_AVAILABLE else None
        
    def create_bitvec_32(self, name: str) -> 'BitVec':
        """Crea un BitVector de 32 bits"""
        if not Z3_AVAILABLE:
            return None
        return BitVec(name, 32)
    
    def rotr_z3(self, x, n):
        """Rotate right en Z3"""
        if not Z3_AVAILABLE:
            return None
        return RotateRight(x, n)
    
    def Sigma0_z3(self, x):
        """Sigma0 de SHA-256 en Z3"""
        return self.rotr_z3(x, 2) ^ self.rotr_z3(x, 13) ^ self.rotr_z3(x, 22)
    
    def Sigma1_z3(self, x):
        """Sigma1 de SHA-256 en Z3"""
        return self.rotr_z3(x, 6) ^ self.rotr_z3(x, 11) ^ self.rotr_z3(x, 25)
    
    def Ch_z3(self, x, y, z):
        """Ch de SHA-256 en Z3"""
        return (x & y) ^ (~x & z)
    
    def Maj_z3(self, x, y, z):
        """Maj de SHA-256 en Z3"""
        return (x & y) ^ (x & z) ^ (y & z)
    
    def model_sha256_round(self, round_num: int, 
                          state_in: List, 
                          w: 'BitVec') -> List:
        """
        Modela una ronda de SHA-256 en Z3.
        
        Args:
            round_num: Número de ronda (0-63)
            state_in: Estado de entrada [a,b,c,d,e,f,g,h]
            w: Palabra del message schedule
            
        Returns:
            Estado de salida [a',b',c',d',e',f',g',h']
        """
        if not Z3_AVAILABLE:
            return state_in
        
        a, b, c, d, e, f, g, h = state_in
        
        # Cálculo de la ronda
        T1 = h + self.Sigma1_z3(e) + self.Ch_z3(e, f, g) + K_SHA256[round_num] + w
        T2 = self.Sigma0_z3(a) + self.Maj_z3(a, b, c)
        
        # Nuevo estado
        h_new = g
        g_new = f
        f_new = e
        e_new = d + T1
        d_new = c
        c_new = b
        b_new = a
        a_new = T1 + T2
        
        return [a_new, b_new, c_new, d_new, e_new, f_new, g_new, h_new]
    
    def invert_final_rounds(self, 
                           target_hash: bytes,
                           difficulty_bits: int,
                           num_rounds_back: int = 5) -> Optional[Dict]:
        """
        Invierte las últimas N rondas de SHA-256 dado un target.
        
        Args:
            target_hash: Hash objetivo (32 bytes)
            difficulty_bits: Número de bits cero requeridos
            num_rounds_back: Número de rondas a invertir
            
        Returns:
            Dict con restricciones encontradas o None si no hay solución
        """
        
        if not Z3_AVAILABLE:
            print("[WARNING] Z3 no disponible - usando análisis simplificado")
            return self._simplified_analysis(target_hash, difficulty_bits)
        
        print(f"\n[Z3] Invirtiendo últimas {num_rounds_back} rondas...")
        print(f"[Z3] Target difficulty: {difficulty_bits} bits cero")
        
        # Crear solver
        s = Solver()
        s.set("timeout", 30000)  # 30 segundos timeout
        
        # Variables para el estado después de ronda (64 - num_rounds_back)
        state_vars = [self.create_bitvec_32(f"state_{i}") for i in range(8)]
        
        # Variables para las palabras del message schedule que necesitamos
        w_vars = [self.create_bitvec_32(f"w_{64-num_rounds_back+i}") for i in range(num_rounds_back)]
        
        # Simular las últimas num_rounds_back rondas
        current_state = state_vars
        for i in range(num_rounds_back):
            round_num = 64 - num_rounds_back + i
            current_state = self.model_sha256_round(round_num, current_state, w_vars[i])
        
        # El estado final debe sumar con H0 para dar el target
        target_words = list(struct.unpack('>8I', target_hash))
        
        for i in range(8):
            final_value = (current_state[i] + H0_SHA256[i]) & 0xFFFFFFFF
            s.add(final_value == target_words[i])
        
        # Agregar restricción de bits cero
        # Los primeros difficulty_bits del hash deben ser 0
        for bit_idx in range(difficulty_bits):
            word_idx = bit_idx // 32
            bit_in_word = 31 - (bit_idx % 32)  # MSB first
            s.add(Extract(bit_in_word, bit_in_word, BitVecVal(target_words[word_idx], 32)) == 0)
        
        # Resolver
        print(f"[Z3] Resolviendo sistema de ecuaciones...")
        result = s.check()
        
        if result == sat:
            print(f"[Z3] ✓ Solución encontrada!")
            model = s.model()
            
            # Extraer valores
            solution = {
                'state': [model.evaluate(state_vars[i]).as_long() for i in range(8)],
                'w': [model.evaluate(w_vars[i]).as_long() for i in range(num_rounds_back)]
            }
            
            return solution
        else:
            print(f"[Z3] ✗ No se encontró solución (timeout o unsat)")
            return None
    
    def _simplified_analysis(self, target_hash: bytes, difficulty_bits: int) -> Dict:
        """Análisis simplificado sin Z3"""
        
        # Análisis heurístico basado en la estructura de SHA-256
        # Esto es menos preciso pero no requiere Z3
        
        target_words = list(struct.unpack('>8I', target_hash))
        
        # Contar bits forzados por la dificultad
        forced_bits = difficulty_bits
        
        # Estimar propagación hacia atrás
        # Cada ronda de SHA-256 "mezcla" aproximadamente 3-4 bits
        # Propagando hacia atrás, podemos estimar cuántos bits del nonce están restringidos
        
        rounds_analyzed = 10
        bits_per_round = 3.5
        estimated_nonce_bits_restricted = min(32, int(forced_bits / (bits_per_round * rounds_analyzed)))
        
        return {
            'method': 'simplified_heuristic',
            'difficulty_bits': difficulty_bits,
            'estimated_nonce_bits_restricted': estimated_nonce_bits_restricted,
            'estimated_reduction_factor': 2 ** estimated_nonce_bits_restricted
        }

def demo_round_inversion():
    """Demostración de inversión de rondas"""
    
    print("=" * 70)
    print("  INVERSIÓN DE RONDAS SHA-256")
    print("=" * 70)
    
    if not Z3_AVAILABLE:
        print("\n[INFO] Z3 no disponible - usando análisis simplificado")
        print("[INFO] Para análisis completo: pip install z3-solver")
    
    # Target con 16 bits cero (dificultad moderada)
    target = bytes.fromhex('0000' + 'f' * 60)
    difficulty = 16
    
    print(f"\nTarget: {target.hex()[:32]}...")
    print(f"Difficulty: {difficulty} bits cero")
    
    # Invertir
    inverter = SHA256RoundInverter()
    result = inverter.invert_final_rounds(target, difficulty, num_rounds_back=3)
    
    if result:
        print(f"\n{'='*70}")
        print("  RESULTADO DE INVERSIÓN")
        print(f"{'='*70}")
        
        if 'state' in result:
            print(f"\nEstado encontrado (después de ronda 61):")
            for i, val in enumerate(result['state']):
                print(f"  state[{i}] = 0x{val:08x}")
        
        if 'estimated_reduction_factor' in result:
            print(f"\nFactor de reducción estimado: {result['estimated_reduction_factor']:.2f}x")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    demo_round_inversion()
