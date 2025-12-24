# -*- coding: utf-8 -*-
"""
TEST DE MEMORIA FRACTAL
=======================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cmfo.universal.fractal_memory import CantorHyperSpace

print("=== TEST MEMORIA FRACTAL DE CANTOR 7D ===")

memory = CantorHyperSpace()

# Texto a guardar
mensaje = "Hola Cosmos 7D"
data = mensaje.encode('utf-8')
print(f"Almacenando: '{mensaje}' ({len(data)} bytes)")

# Guardar en canal 7 (Milnor structure 7)
bits_written = memory.store(data, milnor_channel=7)
print(f"Bits escritos: {bits_written}")

# Recuperar
recuperado_bytes = memory.retrieve(bits_written, milnor_channel=7)
recuperado_str = recuperado_bytes.decode('utf-8')

print(f"Recuperado:  '{recuperado_str}'")

if mensaje == recuperado_str:
    print("\n[SUCCESS] Memoria Fractal Verificada Correctamente.")
    
    # Check capacity metric
    cap = memory.holographic_capacity()
    print(f"Capacidad Holográfica Teórica: {cap:.2f} bits/célula")
else:
    print("\n[FAIL] Error en recuperación.")
    print(f"Esperado: {list(data)}")
    print(f"Obtenido: {list(recuperado_bytes)}")
