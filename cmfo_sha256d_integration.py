
"""
Integración de SHA256d reversible en el pipeline CMFO.
Convierte el hash reversible en un operador cuántico para el autómata CMFO.
"""
from typing import List, Tuple, Dict, Any
import sys
import os

# Ensure bindings path is available
current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, 'bindings', 'python')
sys.path.insert(0, bindings_path)

from cmfo.crypto.sha256d_reversible import ReversibleSHA256

class CMFO_SHA256d_Operator:
    """
    Operador CMFO para SHA256d reversible.
    Encapsula el hash como transformación unitaria en el espacio de estados extendido.
    """
    
    def __init__(self, header_bits: List[int]):
        """
        header_bits: 640 bits del header (80 bytes) como lista de 0/1
        """
        self.header = header_bits
        self.sha = ReversibleSHA256(trace_mode=True)
    
    def apply_forward(self, state):
        """
        Aplica SHA256d forward al estado.
        state: diccionario con registros CMFO
        """
        # Convertir header a bytes
        header_bytes = bits_to_bytes(self.header)
        
        # Ejecutar hash reversible
        hash_result, fractal_ram = self.sha.sha256d_forward_reversible(header_bytes)
        
        # Actualizar estado CMFO
        state['hash_output'] = hash_result
        state['fractal_ram'] = fractal_ram
        
        return state
    
    def apply_uncompute(self, state):
        """
        Aplica SHA256d uncompute al estado.
        """
        if 'hash_output' not in state:
            raise ValueError("Estado no tiene hash_output para descomputar")
        
        self.sha.sha256d_uncompute(state['hash_output'])
        state['hash_output'] = None
        
        return state
    
    def get_observables(self):
        """
        Extrae observables del FractalRAM para análisis CMFO.
        """
        if not self.sha.fractal_ram:
            return []
        
        return self.sha.fractal_ram.collapse()

def bits_to_bytes(bits: List[int]) -> bytes:
    """Convierte lista de bits a bytes."""
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= (bits[i + j] << (7 - j))
        bytes_list.append(byte)
    return bytes(bytes_list)

if __name__ == "__main__":
    print("[CMFO] Integration Module Loaded.")
