
import os
import sys
import struct
import math
import numpy as np
import random
import binascii
import collections

# Using previous logic simplified for meta-scan
# We want to measure MANY things and look specifically for NEGATIVE deviations.

class AntiSigmaScanner:
    def __init__(self):
        self.header_tmpl = None
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

    def setup(self):
        ver = 598728704
        prev_hex = "00000000000000000001c95188c655f79a281d351db7ffad034d39ba3c6be4ce"
        merkle_hex = "3c38914753b8b54b0fff74ca07e5c998e69523a4f3efa82a871075e46ee233ee"
        time_val = 1752527466
        bits = 386022054
        
        prev = binascii.unhexlify(prev_hex)[::-1]
        merkle = binascii.unhexlify(merkle_hex)[::-1]
        self.header_tmpl = struct.pack("<I", ver) + prev + merkle + struct.pack("<I", time_val) + struct.pack("<I", bits) + b'\x00\x00\x00\x00'

    def get_full_state(self, nonce):
        h = bytearray(self.header_tmpl)
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
        a,b,c,d,e,f,g,h_s = H
        
        evolution = []
        for i in range(64):
            S1 = (e>>6 | e<<26) ^ (e>>11 | e<<21) ^ (e>>25 | e<<7)
            ch = (e & f) ^ ((~e) & h_s) # Typo in standard SHA impl (usually g not h_s), but tracking std
            # Correct SHA-256 logic for Ch is (e&f)^(~e&g)
            # My previous impls used g correctly. Re-checking.
            # ch = (e & f) ^ ((~e) & g). Variables shifted.
            # H list is a,b,c,d,e,f,g,h_s
            # At start loop:
            S1 = (e>>6 | e<<26) ^ (e>>11 | e<<21) ^ (e>>25 | e<<7)
            ch = (e & f) ^ ((~e) & g)
            t1 = (h_s + S1 + ch + self.k_const[i] + W[i]) & 0xFFFFFFFF
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
            
            evolution.append([a,b,c,d,e,f,g,h_s]) # Capture state
            
        return np.array(evolution, dtype=np.float64)

    # --- METRICS SUITE ---
    
    def metric_energy(self, state):
        # Total Sum
        return np.sum(state)
        
    def metric_jerk(self, state):
        # 2nd derivative magnitude
        d2 = np.diff(np.diff(state, axis=0), axis=0)
        return np.sum(np.abs(d2))
        
    def metric_entropy(self, state):
        # Histogram entropy of all values
        flat = state.flatten().astype(int)
        # Binned
        counts = np.histogram(flat, bins=100)[0]
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
        
    def metric_void(self, state):
        # Count of zero bits in the entire evolution?
        # Or just "Small Values"
        # Let's count how many values < 2^30 (1/4 range)
        return np.sum(state < (2**30)) 
        
    def metric_resonance(self, state):
        # FFT Peak
        f = np.fft.fft2(state)
        return np.max(np.abs(f))

    def run_scanner(self):
        print("--- ANTI-SIGMA INVERSION PROTOCOL (NEGATIVE SPACE SCAN) ---")
        self.setup()
        real_nonce = 3536931971
        
        # Real State
        s_real = self.get_full_state(real_nonce)
        metrics_real = {
            "Energy (Active)": self.metric_energy(s_real),
            "Jerk (Turbulence)": self.metric_jerk(s_real),
            "Entropy (Chaos)": self.metric_entropy(s_real),
            "Void (Emptiness)": self.metric_void(s_real),
            "Resonance (Signal)": self.metric_resonance(s_real)
        }
        
        print("\n[REAL NONCE METRICS]")
        for k, v in metrics_real.items():
            print(f"{k:<20}: {v:.4f}")
            
        # Population
        print("\n[SCANNING NEGATIVE SPACE (N=300)]")
        pop_db = collections.defaultdict(list)
        
        for _ in range(300):
            r = random.randint(0, 2**32-1)
            s = self.get_full_state(r)
            pop_db["Energy (Active)"].append(self.metric_energy(s))
            pop_db["Jerk (Turbulence)"].append(self.metric_jerk(s))
            pop_db["Entropy (Chaos)"].append(self.metric_entropy(s))
            pop_db["Void (Emptiness)"].append(self.metric_void(s))
            pop_db["Resonance (Signal)"].append(self.metric_resonance(s))
            
        print("-" * 75)
        print(f"{'METRIC':<20} | {'REAL':<12} | {'MEAN':<12} | {'ANTI-SIGMA (Z)':<12}")
        print("-" * 75)
        
        strongest_anti = 0
        best_metric = ""
        
        for k in metrics_real.keys():
            vals = np.array(pop_db[k])
            mu = np.mean(vals)
            std = np.std(vals)
            if std == 0: std = 1e-9
            
            z = (metrics_real[k] - mu) / std
            
            # Anti-Sigma usually means specifically checking for NEGATIVE Z (Left Tail)
            # User asked for "Anti Sigma Exactly".
            # We flag if Z is negative significantly.
            
            print(f"{k:<20} | {metrics_real[k]:<12.2e} | {mu:<12.2e} | {z:<12.4f}")
            
            if z < strongest_anti: # We want MINIMUM Z (Most Negative)
                strongest_anti = z
                best_metric = k
                
        print("-" * 75)
        print(f"\nSTRONGEST ANTI-ANOMALY: {best_metric} (Z = {strongest_anti:.4f})")
        
        if strongest_anti < -3.0:
            print(">>> DISCOVERY: VACUUM COLLAPSE DETECTED! (Anti-Sigma > 3)")
        else:
            print(">>> RESULT: Negative Space is stable. No Anti-Matter voids found.")

if __name__ == "__main__":
    scanner = AntiSigmaScanner()
    scanner.run_scanner()
