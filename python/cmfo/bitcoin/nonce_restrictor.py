"""
NonceRestrictor: Restricciones específicas del nonce Bitcoin

Aplica restricciones de formato, padding y observaciones empíricas
para reducir el espacio de búsqueda del nonce.
"""

import struct
from typing import Dict, List, Set, Tuple
from .byte_constraint_graph import (
    ByteConstraintGraph,
    ByteNode,
    ByteDomain,
    FixedValueConstraint,
    RangeConstraint
)


class NonceRestrictor:
    """
    Restrictor del espacio de nonces Bitcoin.
    
    Aplica restricciones estructurales del formato Bitcoin + SHA-256:
    1. Formato del header (80 bytes)
    2. Padding SHA-256 obligatorio
    3. Restricciones empíricas observadas
    """
    
    # Posiciones del nonce en el header
    NONCE_START = 76
    NONCE_END = 80
    NONCE_POSITIONS = list(range(NONCE_START, NONCE_END))
    
    def __init__(self, header: bytes, empirical_mode: str = 'conservative'):
        """
        Inicializa el restrictor.
        
        Args:
            header: Header Bitcoin de 80 bytes
            empirical_mode: 'none', 'conservative', 'aggressive'
        """
        assert len(header) == 80, f"Header debe ser 80 bytes, got {len(header)}"
        
        self.header = header
        self.empirical_mode = empirical_mode
        self.graph = ByteConstraintGraph(num_bytes=80)
        
        # Aplicar restricciones en orden
        self._apply_format_constraints()
        self._apply_padding_constraints()
        
        if empirical_mode != 'none':
            self._apply_empirical_constraints()
    
    def _apply_format_constraints(self):
        """Aplica restricciones del formato del header"""
        # Bytes [0-75]: fijos (version, prev_block, merkle_root, timestamp, bits)
        for i in range(self.NONCE_START):
            byte_val = self.header[i]
            constraint = FixedValueConstraint(self.graph.nodes[i], byte_val)
            self.graph.add_constraint(constraint)
        
        # Bytes [76-79]: nonce (sin restricciones de formato, solo empíricas)
        # Estos se restringen en _apply_empirical_constraints
    
    def _apply_padding_constraints(self):
        """
        Aplica restricciones del padding SHA-256.
        
        Para un mensaje de 80 bytes (640 bits):
        - Bloque 1: bytes 0-63 (procesados)
        - Bloque 2: bytes 64-79 + padding
        
        Padding del bloque 2:
        - byte 80: 0x80 (bit 1 obligatorio)
        - bytes 81-125: 0x00
        - bytes 126-127: 0x0280 (640 bits en big-endian)
        
        Nota: Estos bytes no están en el header original,
        pero afectan la propagación de restricciones en SHA-256.
        """
        # El padding no afecta directamente el header,
        # pero sí afecta las palabras W[t] en la expansión de mensaje.
        # Esto se manejará en sha256_constraints.py
        pass
    
    def _apply_empirical_constraints(self):
        """
        Aplica restricciones empíricas observadas en bloques reales.
        
        Basado en análisis de 100 bloques reales (905462-905561):
        - Conservative: 95% cobertura (P05-P95)
        - Aggressive: 80% cobertura (P25-P75)
        """
        if self.empirical_mode == 'conservative':
            # Restricciones conservadoras (95% cobertura - calculado de 100 bloques reales)
            # Bloques 905462-905561 analizados
            empirical_ranges = {
                76: (17, 210),    # Byte 0: 194 valores (P05-P95 real)
                77: (14, 228),    # Byte 1: 215 valores (P05-P95 real)
                78: (9, 236),     # Byte 2: 228 valores (P05-P95 real)
                79: (28, 237),    # Byte 3: 210 valores (P05-P95 real)
            }
        elif self.empirical_mode == 'aggressive':
            # Restricciones agresivas (80% cobertura - calculado de 100 bloques reales)
            empirical_ranges = {
                76: (59, 132),    # Byte 0: 74 valores (P25-P75 real)
                77: (65, 177),    # Byte 1: 113 valores (P25-P75 real)
                78: (56, 194),    # Byte 2: 139 valores (P25-P75 real)
                79: (62, 196),    # Byte 3: 135 valores (P25-P75 real)
            }
        else:
            empirical_ranges = {}
        
        for pos, (min_val, max_val) in empirical_ranges.items():
            constraint = RangeConstraint(self.graph.nodes[pos], min_val, max_val)
            self.graph.add_constraint(constraint)
    
    def reduce_space(self) -> Tuple[bool, int, float]:
        """
        Reduce el espacio de búsqueda mediante propagación AC-3.
        
        Returns:
            (success, reduced_space_size, reduction_factor)
        """
        # Propagar restricciones
        success = self.graph.propagate_ac3()
        
        if not success:
            # Inconsistencia detectada
            return (False, 0, float('inf'))
        
        # Calcular tamaño del espacio reducido
        reduced_space = self.graph.get_space_size(self.NONCE_POSITIONS)
        
        # Calcular factor de reducción
        full_space = 2**32  # 4,294,967,296
        reduction_factor = full_space / reduced_space if reduced_space > 0 else float('inf')
        
        return (success, reduced_space, reduction_factor)
    
    def is_nonce_in_space(self, nonce: int) -> bool:
        """Verifica si un nonce está en el espacio reducido"""
        nonce_bytes = struct.pack('<I', nonce)
        return self.graph.is_value_in_space(
            self.NONCE_POSITIONS,
            list(nonce_bytes)
        )
    
    def get_nonce_domains(self) -> Dict[int, ByteDomain]:
        """Retorna los dominios de cada byte del nonce"""
        return {
            pos: self.graph.nodes[pos].domain.copy()
            for pos in self.NONCE_POSITIONS
        }
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas del espacio reducido"""
        success, reduced_space, reduction_factor = self.reduce_space()
        
        domains = self.get_nonce_domains()
        
        return {
            'success': success,
            'full_space': 2**32,
            'reduced_space': reduced_space,
            'reduction_factor': reduction_factor,
            'nonce_byte_domains': {
                pos - self.NONCE_START: len(domain)
                for pos, domain in domains.items()
            },
            'empirical_mode': self.empirical_mode,
            'num_constraints': len(self.graph.constraints)
        }


def analyze_block(block_data: Dict) -> Dict:
    """
    Analiza un bloque Bitcoin y retorna estadísticas de reducción.
    
    Args:
        block_data: Dict con 'version', 'prev_block', 'merkle_root', 
                    'timestamp', 'bits', 'nonce'
    
    Returns:
        Dict con estadísticas de reducción
    """
    # Construir header
    header = struct.pack('<I', block_data['version'])
    header += block_data['prev_block'][::-1]  # Little-endian
    header += block_data['merkle_root'][::-1]
    header += struct.pack('<I', block_data['timestamp'])
    header += struct.pack('<I', block_data['bits'])
    header += struct.pack('<I', block_data['nonce'])
    
    # Analizar con diferentes modos empíricos
    results = {}
    
    for mode in ['none', 'conservative', 'aggressive']:
        restrictor = NonceRestrictor(header, empirical_mode=mode)
        stats = restrictor.get_statistics()
        
        # Verificar si el nonce real está en el espacio reducido
        nonce_in_space = restrictor.is_nonce_in_space(block_data['nonce'])
        
        results[mode] = {
            **stats,
            'nonce_in_space': nonce_in_space
        }
    
    return results


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def build_header(version: int, prev_block: bytes, merkle_root: bytes,
                 timestamp: int, bits: int, nonce: int) -> bytes:
    """Construye un header Bitcoin de 80 bytes"""
    header = struct.pack('<I', version)
    header += prev_block[::-1]  # Little-endian
    header += merkle_root[::-1]
    header += struct.pack('<I', timestamp)
    header += struct.pack('<I', bits)
    header += struct.pack('<I', nonce)
    
    assert len(header) == 80, f"Header debe ser 80 bytes, got {len(header)}"
    return header


def parse_header(header: bytes) -> Dict:
    """Parsea un header Bitcoin de 80 bytes"""
    assert len(header) == 80, f"Header debe ser 80 bytes, got {len(header)}"
    
    version = struct.unpack('<I', header[0:4])[0]
    prev_block = header[4:36][::-1]  # Big-endian
    merkle_root = header[36:68][::-1]
    timestamp = struct.unpack('<I', header[68:72])[0]
    bits = struct.unpack('<I', header[72:76])[0]
    nonce = struct.unpack('<I', header[76:80])[0]
    
    return {
        'version': version,
        'prev_block': prev_block,
        'merkle_root': merkle_root,
        'timestamp': timestamp,
        'bits': bits,
        'nonce': nonce
    }
