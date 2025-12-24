
import os
import sys
import struct
import math
import numpy as np
import random
import zlib
import binascii

class ImperfectionScanner:
    """
    Searches for Topological Defects ("The Pimple").
    Hypothesis: The Valid Nonce is the 'Ugliest', 'Roughest', or 'Least Compressible' state.
    It represents a 'Glitch' in the matrix.
    """
    
    def __init__(self):
        pass

    def get_state_bytes(self, header, nonce):
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
        # Pack into bytes for compression test
        return struct.pack(">8I", *final_state)

    def analyze_imperfection(self, data_bytes):
        """
        1. Incompressibility: Ratio of Compressed/Raw. 
           If > 1.0 (or very high), it's "Ugly" noise.
        2. Roughness: Sum of absolute differences between bytes.
        3. Asymmetry: Distance between first half and reversed second half.
        """
        
        # A. Compression Resistance (The "Snot" Factor - Unstructured)
        compressed = zlib.compress(data_bytes)
        ratio = len(compressed) / len(data_bytes)
        
        # B. Roughness (The "Pimple" Factor - Jagged)
        # Byte-wise jaggedness
        bytes_arr = list(data_bytes)
        diffs = np.abs(np.diff(bytes_arr))
        roughness = np.sum(diffs) / (len(bytes_arr) * 255.0) # Normalize
        
        # C. Asymmetry (The "Imperfect Face")
        half = len(bytes_arr) // 2
        p1 = bytes_arr[:half]
        p2 = bytes_arr[half:][::-1] # Reverse second half
        # Hamming distance / Abs distance
        asym = np.sum(np.abs(np.array(p1) - np.array(p2)))
        asym_score = asym / (half * 255.0)
        
        return ratio, roughness, asym_score

def run_defect_scan():
    print("--- TOPOLOGICAL DEFECT SCANNER (THE PIMPLE) ---")
    print("Hypothesis: The Nonce is the Maximum Imperfection (Chaos Glitch).")
    
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
    
    scanner = ImperfectionScanner()
    
    # Real
    raw_real = scanner.get_state_bytes(header_tmpl, real_nonce)
    rat_real, rou_real, asy_real = scanner.analyze_imperfection(raw_real)
    
    print(f"\n[REAL NONCE DEFECTS]")
    print(f"Incompressibility: {rat_real:.4f} (Higher = Uglier)")
    print(f"Roughness:         {rou_real:.4f} (Higher = Jagged)")
    print(f"Asymmetry:         {asy_real:.4f} (Higher = Imperfect)")
    
    # Population
    print("\n[SMOOTH POPULATION (N=1000)]")
    rats, rous, asys = [], [], []
    
    for _ in range(1000):
        r = random.randint(0, 2**32-1)
        raw = scanner.get_state_bytes(header_tmpl, r)
        rt, ro, ay = scanner.analyze_imperfection(raw)
        rats.append(rt)
        rous.append(ro)
        asys.append(ay)
        
    means = [np.mean(rats), np.mean(rous), np.mean(asys)]
    stds = [np.std(rats), np.std(rous), np.std(asys)]
    
    reals = [rat_real, rou_real, asy_real]
    # We want POSITIVE Sigma (More Defective than average)
    sigmas = [(reals[i] - means[i]) / stds[i] for i in range(3)]
    
    print("-" * 60)
    print(f"{'METRIC':<20} | {'REAL':<10} | {'AVG':<10} | {'SIGMA':<10}")
    print("-" * 60)
    names = ["Incompressibility", "Roughness", "Asymmetry"]
    max_sig = 0
    best_defect = ""
    
    for i in range(3):
        print(f"{names[i]:<20} | {reals[i]:<10.4f} | {means[i]:<10.4f} | {sigmas[i]:<10.2f}")
        if sigmas[i] > max_sig: 
            max_sig = sigmas[i]
            best_defect = names[i]
            
    print("-" * 60)
    
    print(f"\nDOMINANT IMPERFECTION: {best_defect} ({max_sig:.2f} Sigma)")
    
    if max_sig > 3.0:
        print(">>> DISCOVERY: MASSIVE GLITCH DETECTED! (The Pimple Found)")
    else:
        print(">>> RESULT: Defect is within standard tolerances (<3 Sigma).")

if __name__ == "__main__":
    run_defect_scan()
