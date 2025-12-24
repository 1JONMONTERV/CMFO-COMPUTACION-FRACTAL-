
import os
import sys
import struct
import math
import numpy as np
import random
import colorsys
import binascii

class ChromaticTorus:
    """
    Maps SHA-256 State to the 360-degree Color Wheel.
    Treats the 8 words as 8 Major Hues.
    Analyzes Spectral Harmony.
    """
    
    def __init__(self):
        pass

    def get_spectrum(self, header, nonce):
        # 1. Get Final State (8 Words)
        h = bytearray(header)
        h[76:80] = struct.pack("<I", nonce)
        h_bytes = bytes(h)
        input_block = h_bytes + b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
        
        # Fast reconstruct 8 words from chunk 2 processing
        chunk = input_block[64:]
        W = list(struct.unpack(">16I", chunk)) + [0]*48
        
        for i in range(16, 64):
            s0 = (W[i-15]>>7 | W[i-15]<<25) ^ (W[i-15]>>18 | W[i-15]<<14) ^ (W[i-15]>>3)
            s1 = (W[i-2]>>17 | W[i-2]<<15) ^ (W[i-2]>>19 | W[i-2]<<13) ^ (W[i-2]>>10)
            W[i] = (W[i-16] + s0 + W[i-7] + s1) & 0xFFFFFFFF
            
        H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
             
        k_const = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        ]
        
        a,b,c,d,e,f,g,h_s = H
        for i in range(64):
            S1 = (e>>6 | e<<26) ^ (e>>11 | e<<21) ^ (e>>25 | e<<7)
            ch = (e & f) ^ ((~e) & g)
            t1 = (h_s + S1 + ch + k_const[i] + W[i]) & 0xFFFFFFFF
            S0 = (a>>2 | a<<30) ^ (a>>13 | a<<19) ^ (a>>22 | a<<10)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (S0 + maj) & 0xFFFFFFFF
            
            h_s = g
            g = f
            f = e
            e = (d + t1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (t1 + t2) & 0xFFFFFFFF
            
        final_state = [a,b,c,d,e,f,g,h_s]
        
        # Map 32-bit words to Hues (0..1.0)
        # Using top 16 bits for Hue, bottom 16 for Saturation/Value?
        # Simpler: just Hue.
        hues = [ (x / 2**32) for x in final_state ]
        return np.array(hues)

    def analyze_harmony(self, hues):
        """
        1. Spectral Purity: Are the hues distributed evenly (Rainbow) or clustered (Muddy)?
           Entropy of the hue histogram.
        2. Complementary Balance: Does every hue have an opposite (diff ~ 0.5)?
           Sum of pairwise vectors. Ideally = 0 (perfect balance).
        3. Gradient Smoothness: Do adjacent words have close hues?
           Sum of angular differences.
        """
        
        # A. Vector Sum (Color Wheel Center of Mass)
        # If perfectly balanced (rainbow), CoM is near 0.
        # If clustered (monochrome), CoM is near 1.
        angles = hues * 2 * np.pi
        x = np.sum(np.cos(angles))
        y = np.sum(np.sin(angles))
        balance = 1.0 - (np.sqrt(x**2 + y**2) / 8.0) # 1.0 = Perfect Balance
        
        # B. Gradient Smoothness (Flow)
        # Difference between adjacent hues (H1->H2->H3...)
        diffs = np.diff(angles)
        # Correct wrapping
        diffs[diffs > np.pi] -= 2*np.pi
        diffs[diffs < -np.pi] += 2*np.pi
        smoothness = 1.0 / (1.0 + np.sum(np.abs(diffs))) # High = Smooth
        
        # C. Spectral Entropy (Wealth of Color)
        # Bin hues into 12 buckets (Color Wheel)
        hist, _ = np.histogram(hues, bins=12, range=(0,1))
        probs = hist / 8.0
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        
        return balance, smoothness, entropy

def run_chromatic_scan():
    print("--- CHROMATIC TORUS SPECTRUM (LIGHT & COLOR) ---")
    print("Mapping SHA-256 State to Color Wheel...")
    
    # 905561 Setup
    ver = 598728704
    prev_hex = "00000000000000000001c95188c655f79a281d351db7ffad034d39ba3c6be4ce"
    merkle_hex = "3c38914753b8b54b0fff74ca07e5c998e69523a4f3efa82a871075e46ee233ee"
    time_val = 1752527466
    bits = 386022054
    real_nonce = 3536931971
    
    prev = binascii.unhexlify(prev_hex)[::-1]
    merkle = binascii.unhexlify(merkle_hex)[::-1]
    header_tmpl = struct.pack("<I", ver) + prev + merkle + struct.pack("<I", time_val) + struct.pack("<I", bits) + b'\x00\x00\x00\x00'
    
    torus = ChromaticTorus()
    
    # Real
    hues_real = torus.get_spectrum(header_tmpl, real_nonce)
    bal_real, sm_real, ent_real = torus.analyze_harmony(hues_real)
    
    print(f"\n[REAL NONCE SPECTRUM]")
    print(f"Color Balance:   {bal_real:.4f} (Max 1.0)")
    print(f"Gradient Flow:   {sm_real:.4f}")
    print(f"Spectral Entropy:{ent_real:.4f}")
    
    # Population
    print("\n[WHITE NOISE SPECTRUM (N=1000)]")
    bal_pop, sm_pop, ent_pop = [], [], []
    
    for _ in range(1000):
        r = random.randint(0, 2**32-1)
        h = torus.get_spectrum(header_tmpl, r)
        b, s, e = torus.analyze_harmony(h)
        bal_pop.append(b)
        sm_pop.append(s)
        ent_pop.append(e)
        
    means = [np.mean(bal_pop), np.mean(sm_pop), np.mean(ent_pop)]
    stds = [np.std(bal_pop), np.std(sm_pop), np.std(ent_pop)]
    
    reals = [bal_real, sm_real, ent_real]
    sigmas = [(reals[i] - means[i]) / stds[i] for i in range(3)]
    
    print("-" * 60)
    print(f"{'METRIC':<20} | {'REAL':<10} | {'AVG':<10} | {'SIGMA':<10}")
    print("-" * 60)
    names = ["Balance", "Flow", "Entropy"]
    max_sig = 0
    
    for i in range(3):
        print(f"{names[i]:<20} | {reals[i]:<10.4f} | {means[i]:<10.4f} | {sigmas[i]:<10.2f}")
        if abs(sigmas[i]) > max_sig: final_sig = abs(sigmas[i])
            
    print("-" * 60)
    
    if final_sig > 3.0:
        print(">>> DISCOVERY: PERFECT RAINBOW DETECTED!")
    else:
        print(">>> RESULT: Colors are muddy magnitude (<3 Sigma).")

if __name__ == "__main__":
    run_chromatic_scan()
