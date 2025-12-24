
import os
import sys
import struct
import math
import numpy as np
import random
import binascii

class HolographicTorus:
    """
    Recursive Toroidal Analysis.
    Every 32-bit state word at every round is mapped to a Circle (S1).
    The total state is a product of these circles (T^512).
    We look for global phase alignment (Standing Wave).
    """
    
    def __init__(self):
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

    def get_hologram(self, header, nonce):
        h = bytearray(header)
        h[76:80] = struct.pack("<I", nonce)
        h_bytes = bytes(h)
        input_block = h_bytes + b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
        
        chunk = input_block[64:]
        W = list(struct.unpack(">16I", chunk)) + [0]*48
        
        for i in range(16, 64):
            s0 = (W[i-15]>>7 | W[i-15]<<25) ^ (W[i-15]>>18 | W[i-15]<<14) ^ (W[i-15]>>3)
            s1 = (W[i-2]>>17 | W[i-2]<<15) ^ (W[i-2]>>19 | W[i-2]<<13) ^ (W[i-2]>>10)
            W[i] = (W[i-16] + s0 + W[i-7] + s1) & 0xFFFFFFFF
            
        H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
             
        a,b,c,d,e,f,g,h = H
        
        # Store phases of (a..h) at every round t
        # Shape: 64 rounds x 8 words
        hologram = []
        
        for i in range(64):
            S1 = (e>>6 | e<<26) ^ (e>>11 | e<<21) ^ (e>>25 | e<<7)
            ch = (e & f) ^ ((~e) & g)
            t1 = (h + S1 + ch + self.k_const[i] + W[i]) & 0xFFFFFFFF
            S0 = (a>>2 | a<<30) ^ (a>>13 | a<<19) ^ (a>>22 | a<<10)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (S0 + maj) & 0xFFFFFFFF
            
            h = g
            g = f
            f = e
            e = (d + t1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (t1 + t2) & 0xFFFFFFFF
            
            # Map to Phases S1
            phases = [(x / 2**32) * 2 * np.pi for x in [a,b,c,d,e,f,g,h]]
            hologram.append(phases)
            
        return np.array(hologram) # (64, 8)

    def measure_resonance(self, hologram):
        """
        Calculates the "Holographic Resonance" as the Standing Wave ratio.
        
        1. Temporal Resonance: Do phases repeat or harmonically relate across time?
           FFT of each column (word evolution). Peak power.
        2. Spatial Resonance: Do phases align across words (Row coherence)?
           Vector sum of row phasors.
        """
        
        # 1. Spatial Coherence (Row-wise)
        # Sum of complex phasors exp(i*theta) across the 8 columns
        row_phasors = np.sum(np.exp(1j * hologram), axis=1)
        row_mag = np.abs(row_phasors) # Magnitude 0..8
        # Average Spatial Coherence across all 64 rounds
        spatial_coh = np.mean(row_mag)
        
        # 2. Temporal Coherence (Column-wise)
        # Power Spectrum Density of the evolution of each word
        col_ffts = np.fft.fft(hologram, axis=0)
        col_power = np.abs(col_ffts)**2
        # Spectral entropy? Or just max peak?
        # Let's use Max Peak / Total Energy (sharpness of resonance)
        peak_ratios = np.max(col_power, axis=0) / np.sum(col_power, axis=0)
        temporal_coh = np.mean(peak_ratios)
        
        # 3. Micro-Torus Flux (Change)
        # Ideally, we want Smooth Flow? Or Periodic Flow?
        # Let's assume Valid Block = High Structure = High Coherence.
        
        return spatial_coh, temporal_coh

def run_holographic_scan():
    print("--- HOLOGRAPHIC TOROIDAL DECOMPOSITION (S1 x 64 x 8) ---")
    print("Scanning for Standing Waves in the Micro-Structure...")
    
    # 905561
    ver = 598728704
    prev_hex = "00000000000000000001c95188c655f79a281d351db7ffad034d39ba3c6be4ce"
    merkle_hex = "3c38914753b8b54b0fff74ca07e5c998e69523a4f3efa82a871075e46ee233ee"
    time_val = 1752527466
    bits = 386022054
    real_nonce = 3536931971
    
    prev = binascii.unhexlify(prev_hex)[::-1]
    merkle = binascii.unhexlify(merkle_hex)[::-1]
    header_tmpl = struct.pack("<I", ver) + prev + merkle + struct.pack("<I", time_val) + struct.pack("<I", bits) + b'\x00\x00\x00\x00'
    
    holo = HolographicTorus()
    
    # Real
    grid_real = holo.get_hologram(header_tmpl, real_nonce)
    s_real, t_real = holo.measure_resonance(grid_real)
    
    print(f"\n[REAL NONCE HOLOGRAM]")
    print(f"Spatial Coherence:  {s_real:.4f} (Max 8.0)")
    print(f"Temporal Coherence: {t_real:.4f} (Max 1.0)")
    
    # Population
    print("\n[VACUUM FIELD (N=500)]")
    s_pop = []
    t_pop = []
    
    for _ in range(500):
        r = random.randint(0, 2**32-1)
        g = holo.get_hologram(header_tmpl, r)
        s, t = holo.measure_resonance(g)
        s_pop.append(s)
        t_pop.append(t)
        
    s_mu = np.mean(s_pop)
    s_std = np.std(s_pop)
    t_mu = np.mean(t_pop)
    t_std = np.std(t_pop)
    
    s_sigma = (s_real - s_mu) / s_std
    t_sigma = (t_real - t_mu) / t_std
    
    print("-" * 60)
    print(f"{'METRIC':<20} | {'REAL':<10} | {'VACUUM AVG':<10} | {'SIGMA':<10}")
    print("-" * 60)
    print(f"{'Spatial (Space)':<20} | {s_real:<10.4f} | {s_mu:<10.4f} | {s_sigma:<10.2f}")
    print(f"{'Temporal (Time)':<20} | {t_real:<10.4f} | {t_mu:<10.4f} | {t_sigma:<10.2f}")
    print("-" * 60)
    
    max_sig = max(abs(s_sigma), abs(t_sigma))
    print(f"\nMAX HOLOGRAPHIC RESONANCE: {max_sig:.2f} Sigma")
    
    if max_sig > 3.0:
        print(">>> DISCOVERY: HOLOGRAPHIC STANDING WAVE FOUND!")
    else:
        print(">>> RESULT: No standing wave. The hologram is incoherent noise.")

if __name__ == "__main__":
    run_holographic_scan()
