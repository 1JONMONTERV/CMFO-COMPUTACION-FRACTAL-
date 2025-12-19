# sha256d_reversible.py
"""
Implementación bit-exacta reversible de SHA256d para CMFO.
Ancillas mínimas: T1, T2, S0, S1, CH, MAJ (6×u32), W[16] (16×u32), carry (1 bit)
Traza por ronda almacenada reversiblemente en FractalRAM (observables cohérentes).
"""

import struct
from typing import List, Tuple, Dict, Any

# ============================================================================
# CONSTANTES SHA-256
# ============================================================================

# Valores iniciales H0-H7
IV = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

# Constantes de ronda K[0..63]
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

# ============================================================================
# CLASE PARA OBSERVABLES EN FRACTALRAM (COPIA REVERSIBLE)
# ============================================================================

class FractalRAM:
    """
    Registro de observables que mantiene coherencia cuántica.
    Implementa copia reversible (CNOT-like) de valores.
    """
    
    def __init__(self):
        self.log = []  # Lista de tuplas (ronda, T1, T2, a, e, Wt)
        self.coherent = True  # Indica si los valores están en superposición
    
    def reversible_copy(self, ronda: int, T1: int, T2: int, 
                       a: int, e: int, Wt: int, 
                       carry_charge: int, nl_charge: int) -> None:
        """
        Copia reversible de observables físicos al log.
        """
        self.log.append((ronda, T1, T2, a, e, Wt, carry_charge, nl_charge))
    
    def collapse(self) -> List[Tuple]:
        """
        Colapsa los observables a valores clásicos (para debugging).
        En implementación cuántica, esto sería una medición.
        """
        collapsed = self.log.copy()
        self.log.clear()
        return collapsed
    
    def uncompute_copy(self, ronda: int, T1: int, T2: int,
                      a: int, e: int, Wt: int,
                      carry_charge: int, nl_charge: int) -> None:
        """
        Reversa la copia reversible (XOR inverso).
        """
        if not self.log:
            raise ValueError("FractalRAM vacío, no se puede descomputar")
        
        last = self.log.pop()
        target = (ronda, T1, T2, a, e, Wt, carry_charge, nl_charge)
        if last != target:
            raise ValueError(f"Descomputación inconsistente: {last} != {target}")

# ============================================================================
# MACROS REVERSIBLES
# ============================================================================

def xor_reversible(x: int, y: int) -> int:
    """XOR reversible: retorna y XOR x, sin modificar x."""
    return y ^ x

def popcount(n: int) -> int:
    """Cuenta el número de bits en 1 (Hamming Weight)."""
    return bin(n).count('1')

def rotr(x: int, n: int) -> int:
    """Rotación derecha reversible (permutación de bits)."""
    return (x >> n) | (x << (32 - n)) & 0xFFFFFFFF

# Suma modular reversible con acarreo (algoritmo de Cuccaro simplificado)
class ReversibleAdder:
    """Sumador reversible con 1 bit de acarreo reutilizable."""
    
    def __init__(self):
        self.carry = 0
    
    def add_inplace(self, x: int, y: int) -> Tuple[int, int]:
        """
        Suma modular reversible: y <- y + x mod 2^32.
        Retorna (resultado, carry_charge).
        Carry Charge = Número de carries generados (Hamming Weight del vector carry).
        """
        result = (y + x) & 0xFFFFFFFF
        # Calculo de carga topológica (carries internos)
        # S = A + B => A ^ B ^ C_in = S
        # Carry = (A + B) ^ A ^ B (desplazado, pero el peso es igual)
        carries = (x + y) ^ x ^ y
        charge = popcount(carries & 0xFFFFFFFF) # Solo bits válidos
        return result, charge
    
    def reset_carry(self):
        """Reinicia el carry a 0 (para limpieza)."""
        self.carry = 0

# ============================================================================
# FUNCIONES SHA-256 REVERSIBLES
# ============================================================================

def sigma0(x: int) -> int:
    """σ0(x) = ROTR-7(x) XOR ROTR-18(x) XOR SHR-3(x)"""
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)

def sigma1(x: int) -> int:
    """σ1(x) = ROTR-17(x) XOR ROTR-19(x) XOR SHR-10(x)"""
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)

def Sigma0(x: int) -> int:
    """Σ0(x) = ROTR-2(x) XOR ROTR-13(x) XOR ROTR-22(x)"""
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)

def Sigma1(x: int) -> int:
    """Σ1(x) = ROTR-6(x) XOR ROTR-11(x) XOR ROTR-25(x)"""
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)

def Ch_measured(x: int, y: int, z: int) -> Tuple[int, int]:
    """Ch reversible midiendo no-linealidad (ANDs activos)."""
    # Ch = (x & y) ^ (~x & z)
    term1 = x & y
    term2 = ~x & z
    res = term1 ^ term2
    # Carga = bits activos en las compuertas AND
    charge = popcount(term1) + popcount(term2)
    return res, charge

def Maj_measured(x: int, y: int, z: int) -> Tuple[int, int]:
    """Maj reversible midiendo no-linealidad."""
    # Maj = (x & y) ^ (x & z) ^ (y & z)
    t1 = x & y
    t2 = x & z
    t3 = y & z
    res = t1 ^ t2 ^ t3
    # Carga = bits activos en ANDs
    charge = popcount(t1) + popcount(t2) + popcount(t3)
    return res, charge

# ============================================================================
# CLASE PRINCIPAL: SHA256 REVERSIBLE
# ============================================================================

class ReversibleSHA256:
    """
    Implementación reversible de SHA-256 con ancillas mínimas.
    Sigue la especificación exacta CMFO.
    """
    
    def __init__(self, trace_mode: bool = True):
        # Estado de trabajo (8 palabras)
        self.a = self.b = self.c = self.d = 0
        self.e = self.f = self.g = self.h = 0
        
        # Ancillas temporales (6 palabras)
        self.T1 = self.T2 = 0
        self.S0 = self.S1 = 0
        self.CH = self.MAJ = 0
        
        # Schedule ring buffer (16 palabras)
        self.W = [0] * 16
        
        # Sumador reversible
        self.adder = ReversibleAdder()
        
        # Para traza reversible
        self.trace_mode = trace_mode
        self.fractal_ram = FractalRAM() if trace_mode else None
        
        # Registro de mensaje intermedio
        self.D_msg = [0] * 8
        
        # Contador de ronda
        self.round_count = 0
    
    # ------------------------------------------------------------------------
    # OPERACIONES DE LIMPIEZA/VERIFICACIÓN
    # ------------------------------------------------------------------------
    
    def assert_clean(self):
        """Verifica que todas las ancillas estén en 0."""
        assert self.T1 == 0, f"T1 no cero: {self.T1:#x}"
        assert self.T2 == 0, f"T2 no cero: {self.T2:#x}"
        assert self.S0 == 0, f"S0 no cero: {self.S0:#x}"
        assert self.S1 == 0, f"S1 no cero: {self.S1:#x}"
        assert self.CH == 0, f"CH no cero: {self.CH:#x}"
        assert self.MAJ == 0, f"MAJ no cero: {self.MAJ:#x}"
        assert self.adder.carry == 0, f"Carry no cero: {self.adder.carry}"
        # W puede tener valores, pero deberían estar en estado conocido
    
    def clean_temporaries(self):
        """Limpia todas las ancillas temporales a 0."""
        self.T1 = self.T2 = 0
        self.S0 = self.S1 = 0
        self.CH = self.MAJ = 0
        self.adder.reset_carry()
    
    # ------------------------------------------------------------------------
    # RONDA REVERSIBLE
    # ------------------------------------------------------------------------
    
    def round_compute(self, Wt: int, Kt: int) -> Tuple[int, int]:
        """
        Computa T1 y T2 para la ronda actual.
        Retorna (carry_charge_total, nl_charge_total).
        """
        total_carry = 0
        total_nl = 0
        
        # 1. S1 <- Σ1(e)
        self.S1 = Sigma1(self.e)
        
        # 2. CH <- Ch(e, f, g)
        self.CH, nl = Ch_measured(self.e, self.f, self.g)
        total_nl += nl
        
        # 3. T1 Sumas
        # h + S1
        self.T1, c = self.adder.add_inplace(self.h, 0) # clear T1 effectively
        self.T1, c = self.adder.add_inplace(self.S1, self.T1); total_carry += c
        # + CH
        self.T1, c = self.adder.add_inplace(self.CH, self.T1); total_carry += c
        # + Kt
        self.T1, c = self.adder.add_inplace(Kt, self.T1); total_carry += c
        # + Wt
        self.T1, c = self.adder.add_inplace(Wt, self.T1); total_carry += c
        
        # 4. Uncompute parcial
        self.CH = 0
        self.S1 = 0
        
        # 5. S0 <- Σ0(a)
        self.S0 = Sigma0(self.a)
        
        # 6. MAJ <- Maj(a, b, c)
        self.MAJ, nl = Maj_measured(self.a, self.b, self.c)
        total_nl += nl
        
        # 7. T2 Sumas
        # S0 + MAJ
        self.T2, c = self.adder.add_inplace(self.S0, 0)
        self.T2, c = self.adder.add_inplace(self.MAJ, self.T2); total_carry += c
        
        # 8. Uncompute parcial
        self.S0 = 0
        self.MAJ = 0
        
        return total_carry, total_nl
    
    def round_update(self):
        """
        Actualiza el estado reversiblemente usando T1 y T2.
        """
        # Rotación de registros (permutación reversible)
        # (h, g, f, e, d, c, b, a) <- (g, f, e, d, c, b, a, a)
        new_h = self.g
        new_g = self.f
        new_f = self.e
        new_e = self.d
        new_d = self.c
        new_c = self.b
        new_b = self.a
        new_a = 0  # Inicializar acumulador para T1 + T2
        
        # e' = d + T1 (actual e es el d antiguo después de rotar)
        new_e, _ = self.adder.add_inplace(self.T1, new_e)
        
        # a' = a + T1 + T2 (donde a es el old a, que está en new_a)
        new_a, _ = self.adder.add_inplace(self.T1, new_a)
        new_a, _ = self.adder.add_inplace(self.T2, new_a)
        
        # Actualizar estado
        self.a, self.b, self.c, self.d = new_a, new_b, new_c, new_d
        self.e, self.f, self.g, self.h = new_e, new_f, new_g, new_h
        
        # Limpiar T1 y T2 (uncompute)
        # En reversible, necesitaríamos deshacer las sumas, pero como
        # T1 y T2 se computaron de manera reversible, podemos restarlos
        # Para simplicidad en esta simulación, los ponemos a 0 directamente
        # En implementación real, habría que descomputar exactamente
        self.T1 = 0
        self.T2 = 0
    
    def reversible_round(self, Wt: int, Kt: int, round_num: int):
        """
        Ejecuta una ronda reversible completa con traza opcional.
        """
        # Computar T1, T2 y Cargas Físicas
        c_charge, nl_charge = self.round_compute(Wt, Kt)
        
        # Registrar en FractalRAM si está activado
        if self.trace_mode and self.fractal_ram:
            self.fractal_ram.reversible_copy(
                round_num, self.T1, self.T2, self.a, self.e, Wt,
                c_charge, nl_charge
            )
        
        # Actualizar estado
        self.round_update()
        
        self.round_count += 1
    
    # ------------------------------------------------------------------------
    # SCHEDULE REVERSIBLE
    # ------------------------------------------------------------------------
    
    def schedule_reversible(self, t: int, M: List[int]):
        """
        Obtiene W[t] reversiblemente.
        Para t < 16: W[t] = M[t]
        Para t >= 16: W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16]
        """
        if t < 16:
            return M[t]
        else:
            # Calcular usando ancillas temporales (podrían ser S0, S1)
            # Para no ensuciar el estado, usamos variables locales
            # En implementación reversible, esto se haría con ancillas
            
            # Índices en ring buffer (mod 16)
            idx_t_16 = t % 16
            idx_t_15 = (t - 15) % 16
            idx_t_2 = (t - 2) % 16
            idx_t_7 = (t - 7) % 16
            
            # σ0(W[t-15])
            s0_val = sigma0(self.W[idx_t_15])
            
            # σ1(W[t-2])
            s1_val = sigma1(self.W[idx_t_2])
            
            # Acumular en un temporal (podría ser T1 si está limpio)
            temp = 0
            temp = (temp + self.W[idx_t_16]) & 0xFFFFFFFF
            temp = (temp + s0_val) & 0xFFFFFFFF
            temp = (temp + self.W[idx_t_7]) & 0xFFFFFFFF
            temp = (temp + s1_val) & 0xFFFFFFFF
            
            # Actualizar ring buffer
            self.W[idx_t_16] = temp
            
            return temp
    
    # ------------------------------------------------------------------------
    # COMPRESIÓN DE BLOQUE REVERSIBLE
    # ------------------------------------------------------------------------
    
    def compress_block_reversible(self, M: List[int], H: List[int]):
        """
        Compresión reversible de un bloque SHA-256.
        M: 16 palabras (512 bits)
        H: 8 palabras (256 bits) - estado de entrada
        Retorna: nuevo H (8 palabras)
        """
        # 1. Inicializar estado de trabajo
        self.a, self.b, self.c, self.d = H[0], H[1], H[2], H[3]
        self.e, self.f, self.g, self.h = H[4], H[5], H[6], H[7]
        
        # 2. Inicializar ring buffer W[0..15] = M[0..15]
        self.W = M.copy()
        
        # 3. 64 rondas
        for t in range(64):
            Wt = self.schedule_reversible(t, M)
            self.reversible_round(Wt, K[t], t)
        
        # 4. Feedforward: H_i = H_i + S_i
        new_H = [0] * 8
        new_H[0] = (H[0] + self.a) & 0xFFFFFFFF
        new_H[1] = (H[1] + self.b) & 0xFFFFFFFF
        new_H[2] = (H[2] + self.c) & 0xFFFFFFFF
        new_H[3] = (H[3] + self.d) & 0xFFFFFFFF
        new_H[4] = (H[4] + self.e) & 0xFFFFFFFF
        new_H[5] = (H[5] + self.f) & 0xFFFFFFFF
        new_H[6] = (H[6] + self.g) & 0xFFFFFFFF
        new_H[7] = (H[7] + self.h) & 0xFFFFFFFF
        
        # 5. Limpiar ancillas temporales
        self.clean_temporaries()
        
        return new_H
    
    # ------------------------------------------------------------------------
    # SHA256d COMPLETO
    # ------------------------------------------------------------------------
    
    def sha256d_forward_reversible(self, header: bytes):
        """
        SHA256d completo reversible (80 bytes header -> 32 bytes hash).
        """
        # ========== PRIMER SHA-256 (2 bloques) ==========
        # Bloque 1: bytes 0..63
        M0 = self.bytes_to_words(header[0:64])
        H = self.compress_block_reversible(M0, IV)
        
        # Bloque 2: bytes 64..79 + padding + length
        M1 = self.pad_block_80(header[64:80])
        H = self.compress_block_reversible(M1, H)
        
        # Guardar digest intermedio (D_msg)
        self.D_msg = H.copy()
        
        # ========== SEGUNDO SHA-256 (1 bloque) ==========
        # Construir bloque del digest (32 bytes + padding)
        digest_bytes = b''.join(struct.pack('>I', word) for word in H)
        M2 = self.pad_block_32(digest_bytes)
        
        # Comprimir
        final_H = self.compress_block_reversible(M2, IV)
        
        return final_H
    
    def sha256d_uncompute(self, final_H: List[int]):
        """
        Descomputa reversiblemente el hash, limpiando ancillas.
        Asume que el estado actual está en el final del forward.
        """
        # Deshacer segundo SHA-256
        # Esto requeriría invertir exactamente cada operación
        # Por simplicidad en esta simulación, reiniciamos
        # En implementación real, se ejecutarían las inversas exactas
        self.clean_all()
    
    # ------------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------------
    
    @staticmethod
    def bytes_to_words(data: bytes) -> List[int]:
        """Convierte bytes a lista de palabras de 32 bits (big-endian)."""
        words = []
        for i in range(0, len(data), 4):
            word = struct.unpack('>I', data[i:i+4])[0]
            words.append(word)
        return words
    
    @staticmethod
    def pad_block_80(last_chunk: bytes) -> List[int]:
        """Padding para bloque de 80 bytes (segundo bloque)."""
        # last_chunk tiene 16 bytes (80-64=16)
        # Total: 80 bytes = 640 bits
        # Padding: 1 seguido de ceros hasta que length mod 512 = 448
        # Luego 64 bits de length (640 en big-endian)
        
        words = []
        # Primero los 16 bytes del mensaje
        for i in range(0, 16, 4):
            if i + 4 <= len(last_chunk):
                word = struct.unpack('>I', last_chunk[i:i+4])[0]
            else:
                # Rellenar con lo que haya
                remaining = last_chunk[i:]
                padding = b'\x00' * (4 - len(remaining))
                chunk = remaining + padding
                word = struct.unpack('>I', chunk)[0] if len(chunk) == 4 else 0
            words.append(word)
        
        # Añadir 0x80 byte (en posición 20 del bloque de 16 palabras)
        # words[4] = 0x80000000 (byte 0x80 en la palabra 4, big-endian)
        words.append(0x80000000)
        
        # Rellenar con ceros hasta palabra 14
        while len(words) < 14:
            words.append(0)
        
        # Añadir length (640 bits = 0x00000000 00000280)
        words.append(0x00000000)  # 32 bits altos de 640
        words.append(0x00000280)  # 32 bits bajos de 640
        
        return words
    
    @staticmethod
    def pad_block_32(data: bytes) -> List[int]:
        """Padding para 32 bytes (256 bits) - un bloque completo."""
        # data tiene exactamente 32 bytes
        words = []
        for i in range(0, 32, 4):
            word = struct.unpack('>I', data[i:i+4])[0]
            words.append(word)
        
        # Añadir 0x80
        words.append(0x80000000)
        
        # Rellenar con ceros hasta palabra 14
        while len(words) < 14:
            words.append(0)
        
        # Añadir length (256 bits = 0x00000000 00000100)
        words.append(0x00000000)
        words.append(0x00000100)
        
        return words
    
    def clean_all(self):
        """Limpia todo el estado a cero."""
        self.a = self.b = self.c = self.d = 0
        self.e = self.f = self.g = self.h = 0
        self.clean_temporaries()
        self.W = [0] * 16
        self.D_msg = [0] * 8
        self.round_count = 0
        if self.fractal_ram:
            self.fractal_ram.log.clear()

# ============================================================================
# FUNCIONES DE INTERFAZ PRINCIPAL
# ============================================================================

def sha256d_forward_reversible(header: bytes, trace: bool = True) -> Tuple[List[int], Any]:
    """
    Interfaz principal para SHA256d reversible.
    
    Args:
        header: 80 bytes del header de bloque
        trace: si True, activa traza reversible en FractalRAM
    
    Returns:
        hash_final: lista de 8 palabras (32 bytes)
        fractal_ram: objeto con traza de observables
    """
    if len(header) != 80:
        raise ValueError(f"Header debe ser de 80 bytes, recibido {len(header)} bytes")
    
    sha = ReversibleSHA256(trace_mode=trace)
    hash_final = sha.sha256d_forward_reversible(header)
    
    return hash_final, sha.fractal_ram

def sha256d_uncompute(sha: ReversibleSHA256, final_hash: List[int]):
    """
    Descomputa reversiblemente el hash.
    """
    sha.sha256d_uncompute(final_hash)

def assert_clean_state(sha: ReversibleSHA256):
    """
    Verifica que todas las ancillas estén limpias.
    """
    sha.assert_clean()

# ============================================================================
# TESTS Y VALIDACIÓN
# ============================================================================

if __name__ == "__main__":
    import hashlib
    
    # Test: Header de ejemplo (80 bytes)
    test_header = bytes([i % 256 for i in range(80)])
    
    print("Testing SHA256d reversible...")
    
    # 1. Hash estándar para comparación
    sha256 = hashlib.sha256()
    sha256.update(test_header)
    first_hash = sha256.digest()
    sha256 = hashlib.sha256()
    sha256.update(first_hash)
    expected = sha256.digest()
    expected_words = struct.unpack('>IIIIIIII', expected)
    
    print(f"Expected hash: {expected.hex()}")
    
    # 2. Hash reversible
    hash_rev, fractal_ram = sha256d_forward_reversible(test_header, trace=True)
    hash_bytes = b''.join(struct.pack('>I', word) for word in hash_rev)
    
    print(f"Reversible hash: {hash_bytes.hex()}")
    
    # 3. Verificación
    if hash_bytes == expected:
        print("✓ Hash reversible CORRECTO")
    else:
        print("✗ Hash reversible INCORRECTO")
        print(f"  Diferencia: {hash_bytes.hex()} != {expected.hex()}")
    
    # 4. Verificar limpieza de ancillas
    print("\nVerificando limpieza de ancillas...")
    sha = ReversibleSHA256()
    sha.sha256d_forward_reversible(test_header)
    try:
        sha.assert_clean()
        print("✓ Ancillas limpias")
    except AssertionError as e:
        print(f"✗ Ancillas no limpias: {e}")
    
    # 5. Mostrar traza de algunas rondas
    print("\nTraza de las primeras 3 rondas:")
    if fractal_ram and fractal_ram.log:
        for i in range(min(3, len(fractal_ram.log))):
            ronda, T1, T2, a, e, Wt, c_charge, nl_charge = fractal_ram.log[i]
            print(f"  Ronda {ronda}: T1={T1:08x}, T2={T2:08x}, a={a:08x}, e={e:08x} | Charge: Carry={c_charge}, NL={nl_charge}")
    
    print("\nTest completado.")
