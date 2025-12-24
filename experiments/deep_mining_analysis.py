
import os
import sys
import struct
import math
import numpy as np
import random
import csv
import json
import time
import urllib.request
import binascii  # Añadido import faltante

# Asegurar que podemos importar módulos locales si es necesario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class DeepMiningAnalyzer:
    def __init__(self):
        # Constantes SHA-256
        self.k_const = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        ]
        
        # Líneas base (Calibración aproximada)
        self.mu_torus = 5.0
        self.std_torus = 2.0
        self.mu_asym = 0.33
        self.std_asym = 0.05
        self.mu_jerk = 1.2e12
        self.std_jerk = 0.1e12

        self.cache_file = os.path.join(os.path.dirname(__file__), '../data/block_headers_cache.json')
        self.headers_cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.headers_cache, f)

    def fetch_full_header(self, block_hash):
        if block_hash in self.headers_cache:
            data = self.headers_cache[block_hash]
            return binascii.unhexlify(data['header_hex']), data['nonce'], data['height']

        try:
            url = f"https://blockchain.info/rawblock/{block_hash}"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                
            ver = data['ver']
            prev = binascii.unhexlify(data['prev_block'])[::-1]
            merkle = binascii.unhexlify(data['mrkl_root'])[::-1]
            time_val = data['time']
            bits = data['bits']
            nonce = data['nonce']
            
            header = struct.pack("<I", ver) + prev + merkle + struct.pack("<I", time_val) + struct.pack("<I", bits) + b'\x00\x00\x00\x00'
            
            # Guardar en caché
            self.headers_cache[block_hash] = {
                'header_hex': binascii.hexlify(header).decode(),
                'nonce': nonce,
                'height': data['height']
            }
            self.save_cache()
            
            return header, nonce, data['height']
        except Exception as e:
            print(f"Error fetching {block_hash}: {e}")
            return None, None, None

    def analyze_block(self, header, nonce):
        h = bytearray(header)
        h[76:80] = struct.pack("<I", nonce)
        h_bytes = bytes(h)
        input_block = h_bytes + b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
        chunk = input_block[64:]
        try:
            W = list(struct.unpack(">16I", chunk)) + [0]*48
        except struct.error: # Manejo de errores si el chunk no tiene el tamaño correcto
             return 0, 0, 0
        
        for i in range(16, 64):
            s0 = (W[i-15]>>7 | W[i-15]<<25) ^ (W[i-15]>>18 | W[i-15]<<14) ^ (W[i-15]>>3)
            s1 = (W[i-2]>>17 | W[i-2]<<15) ^ (W[i-2]>>19 | W[i-2]<<13) ^ (W[i-2]>>10)
            W[i] = (W[i-16] + s0 + W[i-7] + s1) & 0xFFFFFFFF
            
        H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
             
        a,b,c,d,e,f,g,h_s = H
        trajectory = []
        full_states = []
        
        for i in range(64):
            S1 = (e>>6 | e<<26) ^ (e>>11 | e<<21) ^ (e>>25 | e<<7)
            ch = (e & f) ^ ((~e) & g)
            t1 = (h_s + S1 + ch + self.k_const[i] + W[i]) & 0xFFFFFFFF
            S0 = (a>>2 | a<<30) ^ (a>>13 | a<<19) ^ (a>>22 | a<<10)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (S0 + maj) & 0xFFFFFFFF
            
            h_s = g; g = f; f = e; e = (d + t1) & 0xFFFFFFFF
            d = c; c = b; b = a; a = (t1 + t2) & 0xFFFFFFFF
            
            vec = [(x / 2**32) * 2 * math.pi for x in [a,b,c,d,e,f,g,h_s]]
            trajectory.append(vec)
            full_states.append([a,b,c,d,e,f,g,h_s])
            
        traj_np = np.array(trajectory)
        final_bytes = struct.pack(">8I", a,b,c,d,e,f,g,h_s)
        
        # 1. Torus (Orden)
        grid = np.pi / 2
        dists = np.abs(np.remainder(traj_np, grid) - grid/2)
        score_torus = 1000.0 / (np.sum(dists) + 1.0)
        
        # 2. Asimetría (Caos/Defecto)
        bytes_arr = list(final_bytes)
        half = len(bytes_arr) // 2
        p1 = bytes_arr[:half]
        p2 = bytes_arr[half:][::-1]
        asym = np.sum(np.abs(np.array(p1) - np.array(p2)))
        score_asym = asym / (half * 255.0) # Normalizar 0..1
        
        # 3. Jerk (Turbulencia - AntiSigma)
        d2 = np.diff(np.diff(full_states, axis=0), axis=0)
        score_jerk = np.sum(np.abs(d2))
        
        return score_torus, score_asym, score_jerk

    def calibrate(self):
        print("Calibrando Línea Base con 100 Muestras Aleatorias...")
        t_list, a_list, j_list = [], [], []
        dummy_header = b'\x00'*80
        for _ in range(100):
            r = random.randint(0, 2**32-1)
            t, a, j = self.analyze_block(dummy_header, r)
            t_list.append(t)
            a_list.append(a)
            j_list.append(j)
            
        self.mu_torus = np.mean(t_list)
        self.std_torus = np.std(t_list)
        self.mu_asym = np.mean(a_list)
        self.std_asym = np.std(a_list)
        self.mu_jerk = np.mean(j_list)
        self.std_jerk = np.std(j_list)
        print(f"Base Toro: {self.mu_torus:.2f} +/- {self.std_torus:.2f}")
        print(f"Base Asim: {self.mu_asym:.2f} +/- {self.std_asym:.2f}")

    def run_full_analysis(self, source_csv='bloques_100.csv', output_report='docs/reports/MINING_DEEP_ANALYSIS.md'):
        print(f"Iniciando Análisis Profundo desde {source_csv}...")
        self.calibrate()
        
        try:
            with open(source_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except FileNotFoundError:
            print(f"Error: No se encontró {source_csv}")
            return
            
        results = []
        type_counts = defaultdict(int)

        print("\nProcesando Bloques...")
        print(f"{'ALTURA':<8} | {'TIPO':<8} | {'TORO (Z)':<9} | {'ASIM (Z)':<9} | {'JERK (Z)':<9} | {'TXs':<5}")
        print("-" * 65)

        for i, row in enumerate(rows):
            # Limite de seguridad o procesar todos si se desea
            if i >= 60: break 
            
            h = row.get('hash') or row.get('Hash')
            if not h: continue
            
            header, nonce, height = self.fetch_full_header(h)
            if not header:
                continue

            t, a, j = self.analyze_block(header, nonce)
            
            # Puntuaciones Z
            z_t = (t - self.mu_torus) / (self.std_torus + 1e-9)
            z_a = (a - self.mu_asym) / (self.std_asym + 1e-9)
            z_j = (j - self.mu_jerk) / (self.std_jerk + 1e-9)
            
            # Lógica de Clasificación
            b_type = "RUIDO"
            if z_t > 0.8: # Umbral ajustado
                b_type = "GEO" # Geómetrico
            elif z_a > 0.8:
                b_type = "CAOS" # Caótico/Asimétrico
                
            type_counts[b_type] += 1
            
            results.append({
                'height': height,
                'type': b_type,
                'z_torus': z_t,
                'z_asym': z_a,
                'z_jerk': z_j,
                'tx_count': row.get('tx_count', 'N/A')
            })
            
            print(f"{height:<8} | {b_type:<8} | {z_t:<9.2f} | {z_a:<9.2f} | {z_j:<9.2f} | {row.get('tx_count',''):<5}")
            
        # Generar Reporte
        self.generate_report(results, type_counts, output_report)
        print(f"\nAnálisis completado. Reporte generado en {output_report}")

    def generate_report(self, results, counts, filepath):
        total = len(results)
        if total == 0: return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Expediente de Análisis Minero Profundo (CMFO)\n\n")
            f.write(f"**Fecha:** {time.strftime('%Y-%m-%d')}\n")
            f.write(f"**Total Bloques Analizados:** {total}\n\n")
            
            f.write("## 1. Resumen Ejecutivo\n")
            f.write("Este expediente detalla la clasificación topológica de bloques válida de Bitcoin utilizando métricas fractales de CMFO.\n")
            f.write("Se han identificado patrones distintivos que sugieren dos regímenes de operación en SHA-256: Geométrico (Resonante) y Caótico (Asimétrico).\n\n")
            
            f.write("## 2. Taxonomía de Bloques\n")
            f.write("| Tipo | Descripción | Cantidad | Porcentaje |\n")
            f.write("|---|---|---|---|\n")
            f.write(f"| **GEO (A)** | Alta Resonancia Toroidal. Estructura ordenada. | {counts['GEO']} | {(counts['GEO']/total)*100:.1f}% |\n")
            f.write(f"| **CAOS (B)** | Alta Asimetría. Defecto topológico dominante. | {counts['CAOS']} | {(counts['CAOS']/total)*100:.1f}% |\n")
            f.write(f"| **RUIDO (C)** | Sin firma clara. Fondo estocástico. | {counts['RUIDO']} | {(counts['RUIDO']/total)*100:.1f}% |\n\n")
            
            f.write("## 3. Detalle de Métricas\n")
            f.write("Se analizaron las siguientes métricas profundas:\n")
            f.write("- **Toro (Z-Score)**: Mide la alineación de la trayectoria de hash con la rejilla del Toro 8D.\n")
            f.write("- **Asimetría (Z-Score)**: Mide la desviación Especular del hash final.\n")
            f.write("- **Jerk (Z-Score)**: Mide la 'sacudida' o turbulencia de tercer orden en la computación.\n\n")
            
            f.write("### Tabla de Datos (Muestra)\n")
            f.write("| Altura | Tipo | Toro (Z) | Asim (Z) | Jerk (Z) | TXs |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in results:
                f.write(f"| {r['height']} | {r['type']} | {r['z_torus']:.2f} | {r['z_asym']:.2f} | {r['z_jerk']:.2f} | {r['tx_count']} |\n")
            
            f.write("\n## 4. Conclusiones Preliminares\n")
            if counts['GEO'] > counts['CAOS']:
                f.write("> **Dominancia Geométrica**: La mayoría de los nonces válidos exhiben propiedades de resonancia.\n")
            else:
                f.write("> **Dominancia Caótica**: El sistema favorece soluciones que maximizan la entropía local.\n")

if __name__ == "__main__":
    from collections import defaultdict
    analyzer = DeepMiningAnalyzer()
    analyzer.run_full_analysis()
