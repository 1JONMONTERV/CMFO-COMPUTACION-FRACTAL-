#!/usr/bin/env python3
"""
Análisis de Nonces Reales de Blockchain

Obtiene los nonces completos de los 100 bloques y analiza su distribución
para calibrar las restricciones empíricas del sistema de minería.
"""

import os
import sys
import csv
import json
import struct
import requests
import time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

def get_block_data_from_api(block_height):
    """Obtiene datos completos de un bloque desde blockchain.info API"""
    try:
        url = f"https://blockchain.info/block-height/{block_height}?format=json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'blocks' in data and len(data['blocks']) > 0:
                block = data['blocks'][0]
                return {
                    'height': block_height,
                    'hash': block['hash'],
                    'nonce': block['nonce'],
                    'bits': block['bits'],
                    'timestamp': block['time']
                }
    except Exception as e:
        print(f"Error obteniendo bloque {block_height}: {e}")
    return None

def analyze_nonce_distribution(nonces):
    """Analiza la distribución de nonces byte por byte"""
    
    # Convertir nonces a bytes
    nonce_bytes = []
    for nonce in nonces:
        # Little-endian (como Bitcoin)
        bytes_le = struct.pack('<I', nonce)
        nonce_bytes.append(list(bytes_le))
    
    nonce_bytes = np.array(nonce_bytes)
    
    # Análisis por byte
    analysis = {}
    for byte_idx in range(4):
        byte_values = nonce_bytes[:, byte_idx]
        
        analysis[byte_idx] = {
            'min': int(np.min(byte_values)),
            'max': int(np.max(byte_values)),
            'mean': float(np.mean(byte_values)),
            'median': float(np.median(byte_values)),
            'std': float(np.std(byte_values)),
            'p05': int(np.percentile(byte_values, 5)),
            'p10': int(np.percentile(byte_values, 10)),
            'p25': int(np.percentile(byte_values, 25)),
            'p75': int(np.percentile(byte_values, 75)),
            'p90': int(np.percentile(byte_values, 90)),
            'p95': int(np.percentile(byte_values, 95)),
            'p99': int(np.percentile(byte_values, 99)),
            'unique_values': len(np.unique(byte_values)),
            'distribution': np.bincount(byte_values, minlength=256).tolist()
        }
    
    return analysis

def calculate_optimal_ranges(analysis, coverage_target=0.95):
    """Calcula rangos óptimos para cada byte basado en cobertura deseada"""
    
    ranges = {}
    for byte_idx in range(4):
        stats = analysis[byte_idx]
        
        if coverage_target >= 0.99:
            min_val = stats['min']
            max_val = stats['max']
        elif coverage_target >= 0.95:
            min_val = stats['p05']
            max_val = stats['p95']
        elif coverage_target >= 0.90:
            min_val = stats['p10']
            max_val = stats['p90']
        else:
            min_val = stats['p25']
            max_val = stats['p75']
        
        ranges[byte_idx] = {
            'min': min_val,
            'max': max_val,
            'size': max_val - min_val + 1
        }
    
    return ranges

def main():
    print("=" * 70)
    print("  ANÁLISIS DE NONCES REALES - CALIBRACIÓN CMFO")
    print("=" * 70)
    
    # 1. Cargar lista de bloques
    csv_path = os.path.join(os.path.dirname(__file__), 'bloques_100.csv')
    
    block_heights = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_heights.append(int(row['height']))
    
    print(f"\n[INFO] Bloques a analizar: {len(block_heights)}")
    print(f"[INFO] Rango: {min(block_heights)} - {max(block_heights)}")
    
    # 2. Obtener nonces reales de la API
    print("\n[FASE 1] Obteniendo nonces reales de blockchain...")
    
    nonces = []
    blocks_data = []
    
    for i, height in enumerate(block_heights):
        print(f"  [{i+1}/{len(block_heights)}] Bloque {height}...", end='', flush=True)
        
        block_data = get_block_data_from_api(height)
        if block_data:
            nonces.append(block_data['nonce'])
            blocks_data.append(block_data)
            print(f" ✓ Nonce: 0x{block_data['nonce']:08x}")
        else:
            print(" ✗ FALLO")
        
        # Rate limiting
        time.sleep(0.5)
    
    print(f"\n[INFO] Nonces obtenidos: {len(nonces)}/{len(block_heights)}")
    
    if len(nonces) < 50:
        print("[ERROR] Insuficientes nonces para análisis confiable")
        return
    
    # 3. Análisis estadístico
    print("\n[FASE 2] Analizando distribución de nonces...")
    
    analysis = analyze_nonce_distribution(nonces)
    
    # 4. Mostrar resultados
    print("\n" + "=" * 70)
    print("  DISTRIBUCIÓN POR BYTE")
    print("=" * 70)
    
    for byte_idx in range(4):
        stats = analysis[byte_idx]
        print(f"\nByte {byte_idx}:")
        print(f"  Rango:    [{stats['min']:3d}, {stats['max']:3d}] (0x{stats['min']:02x}, 0x{stats['max']:02x})")
        print(f"  Media:    {stats['mean']:.1f}")
        print(f"  Mediana:  {stats['median']:.1f}")
        print(f"  Std Dev:  {stats['std']:.1f}")
        print(f"  P05-P95:  [{stats['p05']:3d}, {stats['p95']:3d}]")
        print(f"  P10-P90:  [{stats['p10']:3d}, {stats['p90']:3d}]")
        print(f"  Valores únicos: {stats['unique_values']}/256")
    
    # 5. Calcular rangos óptimos
    print("\n" + "=" * 70)
    print("  RANGOS ÓPTIMOS RECOMENDADOS")
    print("=" * 70)
    
    for coverage in [0.99, 0.95, 0.90, 0.80]:
        ranges = calculate_optimal_ranges(analysis, coverage)
        
        total_space = 1
        for r in ranges.values():
            total_space *= r['size']
        
        reduction = (2**32) / total_space
        
        print(f"\nCobertura {coverage*100:.0f}%:")
        for byte_idx in range(4):
            r = ranges[byte_idx]
            print(f"  Byte {byte_idx}: [0x{r['min']:02x}, 0x{r['max']:02x}] ({r['size']} valores)")
        print(f"  Espacio total: {total_space:,}")
        print(f"  Reducción: {reduction:.2f}x")
    
    # 6. Guardar resultados
    output_file = 'nonce_analysis_real.json'
    with open(output_file, 'w') as f:
        json.dump({
            'blocks_analyzed': len(nonces),
            'nonces': [int(n) for n in nonces],
            'analysis': analysis,
            'recommended_ranges': {
                'ultra_conservative_99': calculate_optimal_ranges(analysis, 0.99),
                'conservative_95': calculate_optimal_ranges(analysis, 0.95),
                'balanced_90': calculate_optimal_ranges(analysis, 0.90),
                'aggressive_80': calculate_optimal_ranges(analysis, 0.80)
            }
        }, f, indent=2)
    
    print(f"\n[SUCCESS] Análisis guardado en: {output_file}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
