
import os
import sys
import struct
import math
import numpy as np
import random
import csv
import binascii

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import Fractal Core for state access if needed, or implement lightweight here
from cmfo.crypto.fractal_sha256 import FractalSHA256, FractalWord

class ToroidalGeometry:
    """
    Maps SHA-256 State (8 x 32-bit words) to a 8-Torus (T^8).
    Hypothesis: Valid blocks exhibit 'Rational Resonance' on this torus.
    """
    
    def __init__(self):
        self.scale = 2.0 * np.pi / (2**32) # Map 32-bit int to 0..2pi
        
    def state_to_angles(self, hash_ints):
        # Map 8 integers to 8 angles theta_i
        return np.array([h * self.scale for h in hash_ints])
        
    def compute_flux(self, angles):
        # Flux: Sum of cosines of differences (Phase Coherence)
        # Low Flux = Random Phase
        # High Flux = Aligned Phases
        flux = 0.0
        count = 0
        for i in range(len(angles)):
            for j in range(i+1, len(angles)):
                diff = angles[i] - angles[j]
                flux += np.cos(diff)
                count += 1
        return flux / count # Normalize -1..1
        
    def compute_rationality(self, angles):
        # Check if angles are close to rational multiples of pi (e.g. 0, pi, pi/2, pi/3...)
        # We check specific hard resonances: N * theta ~ 0 (mod 2pi) for small N
        score = 0.0
        for theta in angles:
            # Check for N=1, 2, 3, 4, 5, 6, 8, 12
            max_res = 0.0
            for n in [1, 2, 3, 4, 6, 8]:
                val = n * theta
                # Distance to nearest multiple of 2pi
                # normalized 0..1 (1 = perfect resonance)
                dist = abs(val - round(val / (2*np.pi)) * 2*np.pi)
                res = 1.0 - (dist / np.pi) # Linear decay
                res = max(0, res)**4 # Sharp peak
                if res > max_res: max_res = res
            score += max_res
        return score / 8.0 # Avg resonance
        
    def analyze_block(self, header_hex, nonce):
        # We need the INTERMEDIATE HASH (H1) and FINAL HASH (H2)
        # Deterministic mining suggests the "Pre-Image" has geometry.
        # Let's look at H1 (the result of first SHA pass).
        
        # 1. Reconstruct Header
        # ... logic to get header bytes ...
        # For this script we will assume 'header_bytes' is passed ready
        pass

def reverse_hex(h):
    return binascii.unhexlify(h)[::-1]

def run_torus_analysis():
    print("--- CMFO TOROIDAL GEOMETRY (T^8) ANALYSIS ---")
    print("Hypothesis: Valid Nonces create Standing Waves on the 8-Torus.")
    
    # Target: Block 905561
    ver = 598728704
    prev_hex = "00000000000000000001c95188c655f79a281d351db7ffad034d39ba3c6be4ce"
    merkle_hex = "3c38914753b8b54b0fff74ca07e5c998e69523a4f3efa82a871075e46ee233ee"
    time_val = 1752527466
    bits = 386022054
    real_nonce = 3536931971
    
    prev = reverse_hex(prev_hex)
    merkle = reverse_hex(merkle_hex)
    
    header_tmpl = struct.pack("<I", ver) + prev + merkle + struct.pack("<I", time_val) + struct.pack("<I", bits) + b'\x00\x00\x00\x00'
    
    # Helper to calculate H1 and H2
    def get_hash_states(header, nonce):
        h = bytearray(header)
        h[76:80] = struct.pack("<I", nonce)
        h_bytes = bytes(h)
        
        padded = h_bytes + b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
        chunk1 = padded[:64]
        chunk2 = padded[64:]
        
        w1 = [FractalWord.from_int(struct.unpack(">I", chunk1[i:i+4])[0]) for i in range(0,64,4)]
        w2 = [FractalWord.from_int(struct.unpack(">I", chunk2[i:i+4])[0]) for i in range(0,64,4)]
        
        fsha = FractalSHA256()
        fsha.compress(w1)
        fsha.compress(w2)
        h1_ints = fsha.get_hash() # The H1 state (Mid-point)
        
        # H2
        h1_bytes = b"".join(struct.pack(">I", x) for x in h1_ints)
        pad2 = h1_bytes + b'\x80' + b'\x00'*23 + struct.pack(">Q", 256)
        w_final = [FractalWord.from_int(struct.unpack(">I", pad2[i:i+4])[0]) for i in range(0,64,4)]
        
        fsha2 = FractalSHA256()
        fsha2.compress(w_final)
        h2_ints = fsha2.get_hash()
        
        return h1_ints, h2_ints

    geom = ToroidalGeometry()
    
    # 1. Analyze Real Nonce
    h1_real, h2_real = get_hash_states(header_tmpl, real_nonce)
    
    theta_h1 = geom.state_to_angles(h1_real)
    flux_h1 = geom.compute_flux(theta_h1)
    rat_h1 = geom.compute_rationality(theta_h1)
    
    theta_h2 = geom.state_to_angles(h2_real)
    flux_h2 = geom.compute_flux(theta_h2)
    rat_h2 = geom.compute_rationality(theta_h2)
    
    print("\n[REAL NONCE GEOMETRY]")
    print(f"H1 (Mid-State) Flux: {flux_h1:.4f} | Rationality: {rat_h1:.4f}")
    print(f"H2 (Final)     Flux: {flux_h2:.4f} | Rationality: {rat_h2:.4f}")
    
    # 2. Analyze Random Field
    print("\n[RANDOM FIELD GEOMETRY (N=100)]")
    fluxes_h2 = []
    rats_h2 = []
    
    for _ in range(100):
        r = random.randint(0, 2**32-1)
        _, h2 = get_hash_states(header_tmpl, r)
        t = geom.state_to_angles(h2)
        fluxes_h2.append(geom.compute_flux(t))
        rats_h2.append(geom.compute_rationality(t))
        
    avg_flux = np.mean(fluxes_h2)
    avg_rat = np.mean(rats_h2)
    max_rat = np.max(rats_h2)
    
    print(f"Random Avg Flux: {avg_flux:.4f}")
    print(f"Random Avg Rationality: {avg_rat:.4f}")
    print(f"Random Max Rationality: {max_rat:.4f}")
    
    print("\n[CONCLUSION]")
    sigma_rat = (rat_h2 - avg_rat) / np.std(rats_h2)
    print(f"Rationality Anomaly: {sigma_rat:.2f} Sigma")
    
    if sigma_rat > 3.0:
        print(">>> SUCCESS: TOROIDAL RESONANCE DETECTED! NEW GEOMETRY VALID.")
    else:
        print(">>> RESULT: Toroidal Geometry does not show anomaly > 3 Sigma.")

if __name__ == "__main__":
    run_torus_analysis()
