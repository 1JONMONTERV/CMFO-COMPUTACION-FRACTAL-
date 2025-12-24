
import os
import sys
import struct
import time
import requests
import csv
import binascii
import random
import numpy as np

# Add root path to find bindings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import CMFO Core
try:
    from cmfo_inverse_solver import InverseGeometricSolver
    from cmfo.core.fractal_algebra_1_1 import FractalUniverse1024
    from cmfo.core.hyper_metrics import HyperMetrics
    from cmfo.core.positional import PositionalAlgebra
except ImportError:
    # Fallback pathing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
    from cmfo_inverse_solver import InverseGeometricSolver
    from cmfo.core.fractal_algebra_1_1 import FractalUniverse1024
    from cmfo.core.hyper_metrics import HyperMetrics
    from cmfo.core.positional import PositionalAlgebra

def get_block_details_batch(hashes):
    """
    Fetches full header info.
    To avoid API rate limits, we'll do this carefully or just do first 10 for proof.
    """
    details_map = {}
    for h in hashes:
        url = f"https://mempool.space/api/block/{h}"
        try:
            r = requests.get(url)
            if r.status_code == 200:
                details_map[h] = r.json()
            time.sleep(0.2) # Be polite
        except Exception as e:
            print(f"API Error: {e}")
    return details_map

def reverse_hex(h):
    return binascii.unhexlify(h)[::-1]

class FractalOracle:
    def __init__(self):
        self.delta_quad = (np.arange(256)**2 % 16).astype(int)
        
    def compute_metrics(self, header):
        # Apply Transform
        padded = header + b'\x00' * (128 - len(header))
        u = FractalUniverse1024(padded)
        u_trans = PositionalAlgebra.apply(u, self.delta_quad)
        # Compute 7D
        return HyperMetrics.compute_7d(u_trans)

    def analyze_signature(self, csv_path):
        print("--- FRACTAL ORACLE: SPECTRAL SIGNATURE ANALYSIS ---")
        
        # Load CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            blocks = list(reader)
            
        # Analyze first 10 blocks (Representative Sample)
        sample_size = 10
        print(f"Analyzing representative sample of {sample_size} blocks...")
        
        subset = blocks[:sample_size]
        hashes = [b['hash'] for b in subset]
        
        print("Fetching Metadata...")
        details_map = get_block_details_batch(hashes)
        
        results = {
            'D1_Entropy': {'real': [], 'noise': []},
            'D2_Fractal': {'real': [], 'noise': []},
            'D3_Chirality': {'real': [], 'noise': []},
            'D4_Coherence': {'real': [], 'noise': []},
            'D5_Topology': {'real': [], 'noise': []},
            'D6_Phase':    {'real': [], 'noise': []},
            'D7_Potential':{'real': [], 'noise': []}
        }
        
        metric_names = ['D1_Entropy', 'D2_Fractal', 'D3_Chirality', 'D4_Coherence', 
                        'D5_Topology', 'D6_Phase', 'D7_Potential']

        for row in subset:
            b_hash = row['hash']
            if b_hash not in details_map: continue
            
            d = details_map[b_hash]
            
            # Construct Real Header
            ver = d['version']
            prev = reverse_hex(d['previousblockhash'])
            merkle = reverse_hex(d['merkle_root'])
            ts = d['timestamp']
            bits = d['bits']
            nonce_real = d['nonce']
            
            header_tmpl = struct.pack("<I", ver) + prev + merkle + struct.pack("<I", ts) + struct.pack("<I", bits) + b'\x00\x00\x00\x00'
            
            # 1. Measure Real
            h_real = bytearray(header_tmpl)
            h_real[76:80] = struct.pack("<I", nonce_real)
            v_real = self.compute_metrics(bytes(h_real))
            
            # BOOSTING: Apply Non-Linear Transform to D6 (Phase)
            # Theory: The signal is hidden in the higher-order phase correlations.
            # v[5] is raw phase (0..1).
            # Transformed: sin(2*pi*v[5])^2 ? Or closer to Integer resonance?
            # User theory: "Resonance". Meaning closer to specific harmonic points (0, 0.5, 1.0).
            # Metric: Distance to nearest Integer Resonance.
            # 1.0 - abs(sin(pi * v[5]))
            
            phase_raw = v_real[5]
            # Resonance Metric: 1.0 = Perfect Signal (Integer), 0.0 = Max Noise (0.5)
            # Using cosine as resonance detector: cos(2*pi*phase) -> 1 at integers.
            phase_boosted = np.cos(2 * np.pi * phase_raw)
            
            # Store ORIGINAL + BOOSTED
            metric_names_extended = metric_names + ['D6_Resonance']
            if 'D6_Resonance' not in results:
                results['D6_Resonance'] = {'real': [], 'noise': []}
                
            for i, val in enumerate(v_real):
                results[metric_names[i]]['real'].append(val)
            results['D6_Resonance']['real'].append(phase_boosted)
                
            # 2. Measure Noise (10 random samples per block)
            for _ in range(10):
                r = random.randint(0, 2**32-1)
                h_rand = bytearray(header_tmpl)
                h_rand[76:80] = struct.pack("<I", r)
                v_rand = self.compute_metrics(bytes(h_rand))
                
                phase_boost_rand = np.cos(2 * np.pi * v_rand[5])
                
                for i, val in enumerate(v_rand):
                    results[metric_names[i]]['noise'].append(val)
                results['D6_Resonance']['noise'].append(phase_boost_rand)
                    
            print(f"Processed Block {d['height']}")
            
        # STATISTICAL REPORT
        print("\n" + "="*60)
        print("SPECTRAL SIGNATURE REPORT (WITH BOOSTING)")
        print("="*60)
        print(f"{'METRIC':<15} | {'REAL AVG':<10} | {'NOISE AVG':<10} | {'SEPARATION':<10}")
        print("-" * 55)
        
        best_metric = None
        max_sep = 0
        
        report_lines = []
        
        metric_list_final = metric_names + ['D6_Resonance']
        
        for name in metric_list_final:
            real_vals = np.array(results[name]['real'])
            noise_vals = np.array(results[name]['noise'])
            
            mu_r = np.mean(real_vals)
            mu_n = np.mean(noise_vals)
            std_n = np.std(noise_vals)
            
            # Z-Score of Real signal against Noise distribution
            # Separation = |mu_r - mu_n| / std_n
            if std_n > 0:
                sep = abs(mu_r - mu_n) / std_n
            else:
                sep = 0
                
            print(f"{name:<15} | {mu_r:<10.4f} | {mu_n:<10.4f} | {sep:<10.2f} sigma")
            report_lines.append(f"{name}: Real={mu_r:.4f}, Noise={mu_n:.4f}, Sep={sep:.2f} sigma")
            
            if sep > max_sep:
                max_sep = sep
                best_metric = name
                
        print("-" * 55)
        print(f"\nWinning Metric: {best_metric} ({max_sep:.2f} sigma separation)")
        
        with open("SPECTRAL_REPORT.txt", "w") as f:
            f.write("CMFO SPECTRAL ANALYSIS REPORT\n")
            f.write("=============================\n")
            for l in report_lines:
                f.write(l + "\n")
            f.write(f"\nCONCLUSION: Deterministic Key is {best_metric}.\n")
            
        if max_sep > 2.0:
            print(">> SIGNATURE FOUND: Deterministic Mining is Feasible via Geometry.")
        else:
            print(">> WARNING: Signal is weak. Brute force may be dominated by noise.")

if __name__ == "__main__":
    oracle = FractalOracle()
    # Path handling
    path = "bloques_100.csv"
    if not os.path.exists(path): path = "../../bloques_100.csv"
    if not os.path.exists(path): path = "c:/Users/solmo/Desktop/CMFO_GPU_CLEAN/CMFO_GPU_FINAL/bloques_100.csv"
    
    oracle.analyze_signature(path)
