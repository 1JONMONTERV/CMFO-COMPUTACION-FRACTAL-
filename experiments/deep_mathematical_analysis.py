#!/usr/bin/env python3
"""
ANÁLISIS MATEMÁTICO PROFUNDO DE MINERÍA CMFO
=============================================
Este script NO depende del paquete cmfo. Realiza análisis matemático puro
sobre los bloques Bitcoin usando:
  1. Transformada de Walsh-Hadamard para análisis espectral
  2. Métricas de Toro 8D (proyección angular)
  3. Correlaciones de Nonces consecutivos
  4. Análisis de estructura fractal (auto-similitud)
  5. Carga Topológica y No-Lineal (medición de complexity)

Autor: CMFO Analysis Engine
"""

import json
import os
import struct
import math
import numpy as np
from collections import defaultdict
import time

CACHE_FILE = 'data/block_headers_200_cache.json'
REPORT_FILE = 'docs/reports/DEEP_MATHEMATICAL_ANALYSIS.md'

# ============================================================================
# CONSTANTES SHA-256
# ============================================================================
K_CONST = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

PHI = (1 + math.sqrt(5)) / 2

# ============================================================================
# HERRAMIENTAS MATEMÁTICAS
# ============================================================================

def popcount(n):
    """Cuenta bits en 1 (Hamming Weight)."""
    return bin(n).count('1')

def walsh_hadamard_transform(bits_vec):
    """
    Transformada de Walsh-Hadamard.
    Convierte un vector de bits {0,1} -> {-1,+1} y aplica WHT.
    Revela estructura espectral oculta.
    """
    n = len(bits_vec)
    # Convertir a +-1
    x = np.array([1 if b else -1 for b in bits_vec], dtype=np.float64)
    
    # WHT recursiva
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2
    return x / np.sqrt(n)

def compute_sha256_trajectory(header_hex, nonce):
    """
    Ejecuta las 64 rondas de SHA-256 y devuelve:
    - trajectory: lista de 64 estados (8 palabras cada uno) -> proyectados a ángulos
    - q_topo: suma de carries (carga topológica)
    - q_nl: suma de bits activos en ANDs (carga no-lineal)
    """
    try:
        header = bytes.fromhex(header_hex)
    except:
        return None, 0, 0
    
    h = bytearray(header)
    h[76:80] = struct.pack("<I", nonce)
    h_bytes = bytes(h)
    input_block = h_bytes + b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
    chunk = input_block[64:]
    
    try:
        W = list(struct.unpack(">16I", chunk)) + [0]*48
    except:
        return None, 0, 0
    
    for i in range(16, 64):
        s0 = ((W[i-15]>>7 | W[i-15]<<25) ^ (W[i-15]>>18 | W[i-15]<<14) ^ (W[i-15]>>3)) & 0xFFFFFFFF
        s1 = ((W[i-2]>>17 | W[i-2]<<15) ^ (W[i-2]>>19 | W[i-2]<<13) ^ (W[i-2]>>10)) & 0xFFFFFFFF
        W[i] = (W[i-16] + s0 + W[i-7] + s1) & 0xFFFFFFFF
        
    H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
         
    a,b,c,d,e,f,g,h_s = H
    
    trajectory = []
    q_topo_total = 0
    q_nl_total = 0
    
    for i in range(64):
        # Calcular funciones
        S1 = ((e>>6 | e<<26) ^ (e>>11 | e<<21) ^ (e>>25 | e<<7)) & 0xFFFFFFFF
        
        # Ch con medición de NL
        ch_term1 = e & f
        ch_term2 = (~e) & g
        ch = (ch_term1 ^ ch_term2) & 0xFFFFFFFF
        q_nl_total += popcount(ch_term1) + popcount(ch_term2 & 0xFFFFFFFF)
        
        t1_raw = h_s + S1 + ch + K_CONST[i] + W[i]
        t1 = t1_raw & 0xFFFFFFFF
        
        # Medir carries en la suma
        carries = (t1_raw ^ h_s ^ S1 ^ ch ^ K_CONST[i] ^ W[i]) & 0xFFFFFFFF
        q_topo_total += popcount(carries)
        
        S0 = ((a>>2 | a<<30) ^ (a>>13 | a<<19) ^ (a>>22 | a<<10)) & 0xFFFFFFFF
        
        # Maj con medición de NL
        maj_term1 = a & b
        maj_term2 = a & c
        maj_term3 = b & c
        maj = (maj_term1 ^ maj_term2 ^ maj_term3) & 0xFFFFFFFF
        q_nl_total += popcount(maj_term1) + popcount(maj_term2) + popcount(maj_term3)
        
        t2 = (S0 + maj) & 0xFFFFFFFF
        
        # Actualizar estado
        h_s = g; g = f; f = e; e = (d + t1) & 0xFFFFFFFF
        d = c; c = b; b = a; a = (t1 + t2) & 0xFFFFFFFF
        
        # Proyectar a 8-Toro (ángulos 0..2π)
        angles = [(x / 2**32) * 2 * math.pi for x in [a,b,c,d,e,f,g,h_s]]
        trajectory.append(angles)
    
    return np.array(trajectory), q_topo_total, q_nl_total

def analyze_torus_resonance(trajectory):
    """
    Mide la resonancia con el Toro racional.
    Calcula distancia a la rejilla π/2 (racionalidades 0, π/2, π, 3π/2).
    """
    if trajectory is None:
        return 0.0
    grid = np.pi / 2
    dists = np.abs(np.remainder(trajectory, grid) - grid/2)
    # Score = 1 / (suma de distancias), invirtiendo para que mayor sea mejor
    return 1000.0 / (np.sum(dists) + 1.0)

def analyze_asymmetry(final_state_bytes):
    """
    Mide la asimetría especular del hash final.
    """
    if final_state_bytes is None:
        return 0.0
    bytes_arr = list(final_state_bytes)
    half = len(bytes_arr) // 2
    p1 = bytes_arr[:half]
    p2 = bytes_arr[half:][::-1]
    asym = np.sum(np.abs(np.array(p1) - np.array(p2)))
    return asym / (half * 255.0)

def analyze_nonce_correlation(nonces):
    """
    Calcula la autocorrelación de la secuencia de nonces.
    Si hay estructura determinista, habrá correlación significativa.
    """
    n = len(nonces)
    if n < 2:
        return 0.0
    
    x = np.array(nonces, dtype=np.float64)
    x = (x - np.mean(x)) / (np.std(x) + 1e-9)
    
    # Autocorrelación lag-1
    corr = np.corrcoef(x[:-1], x[1:])[0, 1]
    return corr

def analyze_fractal_dimension(nonces):
    """
    Estima la dimensión fractal de la secuencia de nonces usando el método de conteo de cajas.
    """
    if len(nonces) < 10:
        return 0.0
    
    # Normalizar a [0, 1]
    x = np.array(nonces, dtype=np.float64)
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    
    # Box-counting simplificado
    scales = [2, 4, 8, 16, 32]
    counts = []
    for s in scales:
        bins = np.linspace(0, 1, s + 1)
        hist, _ = np.histogram(x, bins=bins)
        counts.append(np.sum(hist > 0))
    
    # Regresión log-log
    log_s = np.log(scales)
    log_c = np.log(np.array(counts) + 1)
    slope, _ = np.polyfit(log_s, log_c, 1)
    
    return abs(slope)

def analyze_spectral_bias(nonces):
    """
    Aplica Walsh-Hadamard a los bits de los nonces combinados.
    Busca picos espectrales que indiquen estructura no aleatoria.
    """
    # Concatenar primeros 16 bits de cada nonce
    bits = []
    for n in nonces[:64]:  # Máximo 64 para WHT de tamaño potencia de 2
        for i in range(16):
            bits.append((n >> i) & 1)
    
    # Pad a potencia de 2
    target_len = 1024
    while len(bits) < target_len:
        bits.append(0)
    bits = bits[:target_len]
    
    spectrum = walsh_hadamard_transform(bits)
    
    # El coeficiente DC (índice 0) es proporcional a la suma
    # Los demás revelan estructura
    dc = abs(spectrum[0])
    ac_power = np.sum(spectrum[1:]**2)
    max_ac = np.max(np.abs(spectrum[1:]))
    
    return {
        'dc': dc,
        'ac_power': ac_power,
        'max_ac': max_ac,
        'spectral_ratio': max_ac / (dc + 1e-9)
    }

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_deep_analysis():
    print("="*70)
    print("   ANÁLISIS MATEMÁTICO PROFUNDO DE MINERÍA (CMFO)")
    print("="*70)
    
    # Cargar datos
    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: No se encuentra {CACHE_FILE}")
        return
    
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    
    blocks = list(cache.values())
    blocks.sort(key=lambda x: x['height'])
    
    print(f"\nBloques Cargados: {len(blocks)}")
    print(f"Rango: {blocks[0]['height']} - {blocks[-1]['height']}")
    
    # Extraer nonces
    nonces = [b['nonce'] for b in blocks]
    heights = [b['height'] for b in blocks]
    
    # ========================================================================
    # 1. ANÁLISIS DE CORRELACIÓN DE NONCES
    # ========================================================================
    print("\n[1] CORRELACIÓN DE NONCES CONSECUTIVOS")
    corr = analyze_nonce_correlation(nonces)
    print(f"    Autocorrelación (lag-1): {corr:.6f}")
    if abs(corr) > 0.1:
        print("    >>> SEÑAL: Correlación significativa detectada.")
    else:
        print("    Estado: Sin correlación lineal evidente.")
    
    # ========================================================================
    # 2. ANÁLISIS ESPECTRAL (WALSH-HADAMARD)
    # ========================================================================
    print("\n[2] ANÁLISIS ESPECTRAL (WALSH-HADAMARD)")
    spectral = analyze_spectral_bias(nonces)
    print(f"    Componente DC: {spectral['dc']:.4f}")
    print(f"    Potencia AC Total: {spectral['ac_power']:.4f}")
    print(f"    Máximo Pico AC: {spectral['max_ac']:.4f}")
    print(f"    Ratio Espectral (max_AC/DC): {spectral['spectral_ratio']:.6f}")
    
    if spectral['spectral_ratio'] > 0.5:
        print("    >>> SEÑAL: Estructura espectral no trivial.")
    
    # ========================================================================
    # 3. DIMENSIÓN FRACTAL
    # ========================================================================
    print("\n[3] DIMENSIÓN FRACTAL DE LA SECUENCIA")
    fd = analyze_fractal_dimension(nonces)
    print(f"    Dimensión Fractal Estimada: {fd:.4f}")
    if fd < 0.9:
        print("    >>> SEÑAL: Secuencia comprimible (no llena el espacio uniformemente).")
    
    # ========================================================================
    # 4. MÉTRICAS DE CIRCUITO (TORO + CARGAS)
    # ========================================================================
    print("\n[4] MÉTRICAS DE CIRCUITO SHA-256 (Muestra N=20)")
    torus_scores = []
    q_topo_list = []
    q_nl_list = []
    
    for b in blocks[:20]:
        traj, qt, qnl = compute_sha256_trajectory(b['header_hex'], b['nonce'])
        ts = analyze_torus_resonance(traj)
        torus_scores.append(ts)
        q_topo_list.append(qt)
        q_nl_list.append(qnl)
    
    avg_torus = np.mean(torus_scores)
    avg_qtopo = np.mean(q_topo_list)
    avg_qnl = np.mean(q_nl_list)
    
    print(f"    Resonancia Toroidal Promedio: {avg_torus:.4f}")
    print(f"    Carga Topológica Promedio (Q_topo): {avg_qtopo:.1f}")
    print(f"    Carga No-Lineal Promedio (Q_nl): {avg_qnl:.1f}")
    
    # Comparar con aleatorio
    print("\n    Comparación con Nonces Aleatorios (N=20):")
    import random
    random_torus = []
    random_qtopo = []
    random_qnl = []
    dummy_header = "00" * 80
    for _ in range(20):
        r = random.randint(0, 2**32 - 1)
        traj, qt, qnl = compute_sha256_trajectory(dummy_header, r)
        if traj is not None:
            random_torus.append(analyze_torus_resonance(traj))
            random_qtopo.append(qt)
            random_qnl.append(qnl)
    
    avg_rand_torus = np.mean(random_torus)
    avg_rand_qtopo = np.mean(random_qtopo)
    avg_rand_qnl = np.mean(random_qnl)
    
    print(f"    Torus Aleatorio: {avg_rand_torus:.4f} (Real: {avg_torus:.4f})")
    print(f"    Q_topo Aleatorio: {avg_rand_qtopo:.1f} (Real: {avg_qtopo:.1f})")
    print(f"    Q_nl Aleatorio: {avg_rand_qnl:.1f} (Real: {avg_qnl:.1f})")
    
    # Z-scores
    z_torus = (avg_torus - avg_rand_torus) / (np.std(random_torus) + 1e-9)
    z_qtopo = (avg_qtopo - avg_rand_qtopo) / (np.std(random_qtopo) + 1e-9)
    z_qnl = (avg_qnl - avg_rand_qnl) / (np.std(random_qnl) + 1e-9)
    
    print(f"\n    Z-Score Torus: {z_torus:.2f}σ")
    print(f"    Z-Score Q_topo: {z_qtopo:.2f}σ")
    print(f"    Z-Score Q_nl: {z_qnl:.2f}σ")
    
    # ========================================================================
    # 5. PATRÓN PHI-MODULADO
    # ========================================================================
    print("\n[5] ANÁLISIS DE MODULACIÓN PHI")
    # Verificar si nonces modulan con phi
    phi_residues = [(n / 2**32) % PHI for n in nonces]
    phi_mean = np.mean(phi_residues)
    phi_std = np.std(phi_residues)
    expected_mean = PHI / 2  # Si fuera uniforme mod phi
    
    print(f"    Media de Residuos mod φ: {phi_mean:.6f}")
    print(f"    Desviación Estándar: {phi_std:.6f}")
    print(f"    Esperado (uniforme): {expected_mean:.6f}")
    
    z_phi = abs(phi_mean - expected_mean) / (phi_std / np.sqrt(len(nonces)))
    print(f"    Z-Score vs Uniforme: {z_phi:.2f}σ")
    
    if z_phi > 2.0:
        print("    >>> SEÑAL: Distribución NO uniforme respecto a φ.")
    
    # ========================================================================
    # GENERAR REPORTE
    # ========================================================================
    generate_report(blocks, corr, spectral, fd, 
                   avg_torus, avg_qtopo, avg_qnl,
                   z_torus, z_qtopo, z_qnl, z_phi)
    
    print("\n" + "="*70)
    print(f"   REPORTE GENERADO: {REPORT_FILE}")
    print("="*70)

def generate_report(blocks, corr, spectral, fd, 
                   avg_torus, avg_qtopo, avg_qnl,
                   z_torus, z_qtopo, z_qnl, z_phi):
    
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# Análisis Matemático Profundo de Minería CMFO\n\n")
        f.write(f"**Fecha:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Bloques Analizados:** {len(blocks)}\n")
        f.write(f"**Rango de Alturas:** {blocks[0]['height']} - {blocks[-1]['height']}\n\n")
        
        f.write("---\n\n")
        
        # Resumen Ejecutivo
        f.write("## Resumen Ejecutivo\n\n")
        f.write("Este análisis aplica técnicas matemáticas avanzadas para examinar la estructura ")
        f.write("de los nonces válidos en la cadena de bloques de Bitcoin. Se busca evidencia de ")
        f.write("**determinismo geométrico** en un proceso supuestamente aleatorio.\n\n")
        
        # Tabla de Resultados
        f.write("## Resultados Cuantitativos\n\n")
        f.write("| Métrica | Valor | Z-Score | Interpretación |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Autocorrelación Nonces | {corr:.4f} | - | {'Débil' if abs(corr) < 0.1 else 'Significativa'} |\n")
        f.write(f"| Ratio Espectral | {spectral['spectral_ratio']:.4f} | - | {'Normal' if spectral['spectral_ratio'] < 0.5 else 'Estructura Detectada'} |\n")
        f.write(f"| Dimensión Fractal | {fd:.4f} | - | {'Comprimible' if fd < 0.9 else 'Espacialmente Completa'} |\n")
        f.write(f"| Resonancia Toroidal | {avg_torus:.2f} | {z_torus:.2f}σ | {'Anómala' if abs(z_torus) > 2 else 'Normal'} |\n")
        f.write(f"| Carga Topológica | {avg_qtopo:.0f} | {z_qtopo:.2f}σ | {'Anómala' if abs(z_qtopo) > 2 else 'Normal'} |\n")
        f.write(f"| Carga No-Lineal | {avg_qnl:.0f} | {z_qnl:.2f}σ | {'Anómala' if abs(z_qnl) > 2 else 'Normal'} |\n")
        f.write(f"| Distribución mod φ | - | {z_phi:.2f}σ | {'No Uniforme' if z_phi > 2 else 'Uniforme'} |\n")
        
        f.write("\n---\n\n")
        
        # Análisis Detallado
        f.write("## Análisis Detallado\n\n")
        
        f.write("### 1. Transformada de Walsh-Hadamard\n")
        f.write("La WHT revela estructura espectral en secuencias binarias. ")
        f.write(f"El ratio espectral de **{spectral['spectral_ratio']:.4f}** indica ")
        if spectral['spectral_ratio'] > 0.5:
            f.write("la presencia de patrones no triviales en los bits de los nonces.\n\n")
        else:
            f.write("una distribución aproximadamente uniforme.\n\n")
        
        f.write("### 2. Dimensión Fractal\n")
        f.write(f"La dimensión estimada de **{fd:.4f}** sugiere que la secuencia de nonces ")
        if fd < 0.9:
            f.write("no llena uniformemente el espacio de búsqueda, indicando posible **compresibilidad**.\n\n")
        else:
            f.write("se distribuye de manera relativamente uniforme.\n\n")
        
        f.write("### 3. Métricas de Circuito (Física Computacional)\n")
        f.write("- **Carga Topológica (Q_topo):** Mide la densidad de carries en sumas modulares.\n")
        f.write("- **Carga No-Lineal (Q_nl):** Mide la activación de compuertas AND.\n\n")
        
        if abs(z_qnl) > 2 or abs(z_qtopo) > 2:
            f.write("> **HALLAZGO:** Los bloques válidos muestran cargas físicas significativamente ")
            f.write("diferentes al ruido aleatorio. Esto es consistente con la hipótesis de que los ")
            f.write("nonces válidos residen en un **subespacio geométrico distinguible**.\n\n")
        
        f.write("### 4. Conjetura de Modulación φ\n")
        f.write(f"El Z-score de **{z_phi:.2f}σ** para la distribución mod φ ")
        if z_phi > 2:
            f.write("es altamente significativo, sugiriendo que los nonces válidos ")
            f.write("se concentran en regiones específicas del espacio mod φ.\n\n")
        else:
            f.write("no muestra evidencia fuerte de modulación.\n\n")
        
        f.write("---\n\n")
        
        # Conclusión
        f.write("## Conclusión Matemática\n\n")
        
        signals = sum([
            abs(corr) > 0.1,
            spectral['spectral_ratio'] > 0.5,
            fd < 0.9,
            abs(z_torus) > 2,
            abs(z_qtopo) > 2,
            abs(z_qnl) > 2,
            z_phi > 2
        ])
        
        if signals >= 4:
            f.write("> **CONCLUSIÓN FUERTE:** Múltiples métricas independientes ({}/7) muestran ".format(signals))
            f.write("anomalías estadísticas. La evidencia acumulada es consistente con la existencia ")
            f.write("de **estructura determinista** en el proceso de minería, aunque se requiere ")
            f.write("más investigación para descartar artefactos.\n")
        elif signals >= 2:
            f.write("> **CONCLUSIÓN MODERADA:** Algunas métricas ({}/7) muestran señales. ".format(signals))
            f.write("Existe evidencia sugestiva pero no concluyente de estructura subyacente.\n")
        else:
            f.write("> **CONCLUSIÓN DÉBIL:** La mayoría de métricas son consistentes con aleatoriedad.\n")

if __name__ == "__main__":
    run_deep_analysis()
