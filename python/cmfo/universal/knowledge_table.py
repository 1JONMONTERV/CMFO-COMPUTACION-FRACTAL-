# -*- coding: utf-8 -*-
"""
TABLA DE CONOCIMIENTO PROCEDURAL INFINITA
=========================================

Implementación del núcleo de la arquitectura "Total Knowledge".
Utiliza hashing fractal (Phi) para mapear conceptos semánticos a 
direcciones geométricas en el Hiper-Espacio de Cantor.

Capacidades:
1. Universal Fractal Hash (Concepto -> R7)
2. Almacenamiento distribuido en capas de Milnor.
3. Recuperación holográfica.
"""

import hashlib
import numpy as np
from .constants import PHI
from .octonion_algebra import Octonion
from .fractal_memory import CantorHyperSpace

class UniversalFractalHash:
    """
    Sistema de hashing determinista que mapea strings arbitrarios 
    a coordenadas en el espacio octoniónico S7.
    """
    
    def __init__(self):
        self.phi = PHI
        
    def hash_concept(self, concept_str):
        """
        Genera un 'hash fractal' para un concepto.
        Retorna:
        - coords: Coordenadas 7D en S7
        - milnor_layer: Capa de Milnor (0-27)
        - energy: Nivel de energía (para prioridad)
        """
        # 1. SHA-256 base para entropía inicial
        sha = hashlib.sha256(concept_str.encode('utf-8')).digest()
        # Asegurar que la semilla esté en rango válido para numpy (32 bits)
        seed = int.from_bytes(sha[:8], 'big') % (2**32)
        
        # 2. Expansión Fractal via Phi
        # Usamos la semilla para generar coordenadas deterministas
        np.random.seed(seed)
        
        # Generar vector 7D en la esfera unitaria
        coords = np.random.randn(7)
        coords /= np.linalg.norm(coords)
        
        # 3. Determinar capa de Milnor basada en resonancia Phi
        # layer = (sum(bytes) * Phi) mod 28
        byte_sum = sum(sha)
        milnor_layer = int((byte_sum * self.phi) % 28)
        
        # 4. Energía del concepto (importancia)
        energy = float(int.from_bytes(sha[8:12], 'big')) / (2**32)
        
        return {
            'coords': coords,
            'milnor_layer': milnor_layer,
            'energy': energy,
            'raw_hash': sha.hex()
        }

class ProceduralKnowledgeTable:
    """
    Tabla de conocimiento auto-expansiva.
    """
    
    def __init__(self):
        self.memory = CantorHyperSpace()
        self.hasher = UniversalFractalHash()
        # Índice inverso simple para demo (en producción sería distribuido)
        self.index = {} 
        
    def learn(self, concept, definition):
        """
        Aprende un nuevo concepto y lo almacena en el espacio fractal.
        """
        # 1. Hash del concepto (Key)
        meta = self.hasher.hash_concept(concept)
        layer = meta['milnor_layer']
        
        # 2. Codificar definición (Value)
        data = definition.encode('utf-8')
        
        # 3. Almacenar en la capa de Milnor correspondiente
        # Usamos el memory.store pero necesitamos vincularlo a la coordenada exacta
        # Por ahora, usamos la abstracción de canales de la memoria
        address_bits = self.memory.store(data, milnor_channel=layer)
        
        # 4. Actualizar índice
        self.index[concept] = {
            'layer': layer,
            'address_start': 0, # Simplificación: asumimos append-only por canal
            'length': len(data) * 8, # bits
            'meta': meta
        }
        
        return meta
        
    def recall(self, concept):
        """
        Recupera el conocimiento asociado a un concepto.
        """
        if concept not in self.index:
            return None
            
        entry = self.index[concept]
        layer = entry['layer']
        length = entry['length']
        
        # Recuperar de la memoria fractal
        # Nota: En la implementación real de FractalMemoryCell necesitaríamos
        # soporte para acceso aleatorio o cursors. Por ahora recuperamos el último.
        data_bytes = self.memory.retrieve(length, milnor_channel=layer)
        
        return data_bytes.decode('utf-8')

    def find_related(self, concept, n=5):
        """
        Encuentra conceptos relacionados geométricamente (distancia en S7).
        """
        if concept not in self.index:
            return []
            
        target_coords = self.index[concept]['meta']['coords']
        
        relations = []
        for other_concept, entry in self.index.items():
            if other_concept == concept:
                continue
                
            other_coords = entry['meta']['coords']
            
            # Distancia geodésica en S7: arccos(dot_product)
            dot = np.dot(target_coords, other_coords)
            # Clip para evitar errores numéricos fuera de [-1, 1]
            dot = np.clip(dot, -1.0, 1.0)
            dist = np.arccos(dot)
            
            relations.append((other_concept, dist))
            
        # Ordenar por cercanía
        relations.sort(key=lambda x: x[1])
        return relations[:n]

