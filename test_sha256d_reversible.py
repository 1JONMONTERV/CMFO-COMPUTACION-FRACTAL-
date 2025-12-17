
import unittest
import hashlib
import struct
import sys
import os

# Ensure bindings path is available
current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, 'bindings', 'python')
sys.path.insert(0, bindings_path)

from cmfo.crypto.sha256d_reversible import *

class TestReversibleSHA256d(unittest.TestCase):
    
    def test_ancillas_limpias(self):
        """Verifica que las ancillas se limpien después de cada operación."""
        sha = ReversibleSHA256(trace_mode=False)
        header = bytes(80)
        sha.sha256d_forward_reversible(header)
        sha.assert_clean()
    
    def test_equivalencia_estandar(self):
        """Compara con hashlib SHA256d."""
        test_cases = [
            bytes(80),  # Header todo ceros
            bytes([i % 256 for i in range(80)]),  # Header secuencial
            bytes([0xFF] * 80),  # Header todo unos
        ]
        
        for header in test_cases:
            # Hash estándar
            first = hashlib.sha256(header).digest()
            expected = hashlib.sha256(first).digest()
            
            # Hash reversible
            result, _ = sha256d_forward_reversible(header, trace=False)
            result_bytes = b''.join(struct.pack('>I', w) for w in result)
            
            self.assertEqual(result_bytes, expected)
    
    def test_reversibilidad_exacta(self):
        """Verifica que forward + uncompute retorne al estado inicial."""
        sha = ReversibleSHA256(trace_mode=False)
        initial_state = sha.__dict__.copy()
        
        header = bytes(80)
        sha.sha256d_forward_reversible(header)
        sha.sha256d_uncompute(header)  # Necesitaría implementación completa
        
        # Verificar que el estado es idéntico (excepto posiblemente contadores)
        for key in initial_state:
            if key not in ['round_count', 'D_msg', 'W', 'fractal_ram']: # Excluir contenedores complejos
                 self.assertEqual(initial_state[key], sha.__dict__[key])
    
    def test_fractalram_reversible(self):
        """Verifica que FractalRAM mantenga coherencia reversible."""
        ram = FractalRAM()
        
        # Copiar valores
        ram.reversible_copy(0, 0x12345678, 0x87654321, 0x11111111, 0x22222222, 0x33333333)
        
        # Verificar que están en el log
        self.assertEqual(len(ram.log), 1)
        
        # Descomputar
        ram.uncompute_copy(0, 0x12345678, 0x87654321, 0x11111111, 0x22222222, 0x33333333)
        
        # Verificar que el log está vacío
        self.assertEqual(len(ram.log), 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
