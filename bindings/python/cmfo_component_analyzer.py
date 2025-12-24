
import os
import sys
import struct
import math
import numpy as np
import binascii
import random

def entropy(data):
    """Calculates Shannon Entropy of a byte sequence."""
    if not data: return 0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def mutual_information_bits(a_int, b_int):
    """
    Measures bitwise correlation.
    Normalized Mutual Information?
    Simple Metric: Hamming Weight of (A AND B) / Expected.
    If random, P(1)=0.5, so P(A&B)=0.25. 
    Ratio > 1.0 means correlation (bits tend to align).
    """
    intersection = a_int & b_int
    weight = bin(intersection).count('1')
    return weight / 8.0 # Expected is 32 * 0.25 = 8 bits for 32-bit correlation check?
    # Wait, simple ratio. 
    # Let's return raw weight vs expected 8 for 32-bit integer overlap.
    # Actually, let's normalize by 32.
    return weight / 32.0 

def symmetry_score(a_bytes, b_bytes):
    """
    Checks if A is a palindrome/mirror of B or similar structure.
    Simple XOR distance?
    """
    # Pad to match length
    l = max(len(a_bytes), len(b_bytes))
    a = a_bytes.ljust(l, b'\x00')
    b = b_bytes.ljust(l, b'\x00')
    
    # XOR
    diff = bytearray([x ^ y for x, y in zip(a, b)])
    # Low Hamming weight = high symmetry
    w = sum(bin(x).count('1') for x in diff)
    return 1.0 - (w / (l * 8))

class ComponentAnalyzer:
    def __init__(self):
        pass

    def reverse_hex(self, h):
        return binascii.unhexlify(h)[::-1]

    def analyze_block(self, block_data):
        print(f"\n[ANALYZING BLOCK {block_data['height']} COMPONENTS]")
        
        # 1. Component Extraction
        ver = struct.pack("<I", block_data['ver'])
        prev = self.reverse_hex(block_data['prev'])
        merkle = self.reverse_hex(block_data['merkle'])
        time_b = struct.pack("<I", block_data['time'])
        bits_b = struct.pack("<I", block_data['bits'])
        nonce_b = struct.pack("<I", block_data['nonce'])
        
        components = {
            'Ver': ver,
            'Prev': prev,
            'Merkle': merkle,
            'Time': time_b,
            'Bits': bits_b,
            'Nonce': nonce_b
        }
        
        # 2. Internal Entropy
        print("--- INTERNAL ENTROPY (Complexity) ---")
        for name, data in components.items():
            e = entropy(data)
            max_e = 8.0 # Max 8 bits/byte
            print(f"{name:<10} | Entropy: {e:.4f} bits/byte")

        # 3. Correlations (Mutual Information with Nonce)
        print("\n--- NONCE SYMMETRY SCAN ---")
        nonce_int = block_data['nonce']
        
        # Prepare integers for correlation (Taking first 4 bytes of 32-byte fields)
        merkle_int = struct.unpack("<I", merkle[:4])[0]
        prev_int = struct.unpack("<I", prev[:4])[0]
        time_int = block_data['time']
        bits_int = block_data['bits']
        ver_int = block_data['ver']
        
        targets = {
            'Ver': ver_int,
            'Prev(Head)': prev_int,
            'Merkle(Head)': merkle_int,
            'Time': time_int,
            'Bits': bits_int
        }
        
        print(f"{'TARGET':<15} | {'BIT OVERLAP':<10} | {'EXPECTED':<10} | {'ANOMALY'}")
        print("-" * 55)
        
        for name, val in targets.items():
            overlap = bin(nonce_int & val).count('1')
            expected = 8.0 # 32 bits * 0.25 (P(1)*P(1) approx) 
            # Assuming ~50% density. Let's verify density.
            d_nonce = bin(nonce_int).count('1') / 32.0
            d_targ = bin(val).count('1') / 32.0
            expected_refined = 32 * d_nonce * d_targ
            
            sigma = (overlap - expected_refined) / math.sqrt(expected_refined) if expected_refined > 0 else 0
            
            print(f"{name:<15} | {overlap:<10} | {expected_refined:<10.2f} | {sigma:<.2f} Sigma")

        # 4. Mirror Symmetry (Does Nonce ~ Reverse(Merkle)?)
        # Checking XOR distance between Nonce and Slices of Merkle
        print("\n--- MERKLE-NONCE MIRROR SEARCH ---")
        min_dist = 256
        best_offset = -1
        
        # Merkle is 32 bytes. Nonce is 4 bytes.
        # Slide Nonce over Merkle
        nonce_bytes = components['Nonce']
        merkle_bytes = components['Merkle']
        
        for i in range(29): # 0..28
            slice_m = merkle_bytes[i:i+4]
            # Hamming Dist
            dist = sum(bin(x ^ y).count('1') for x, y in zip(nonce_bytes, slice_m))
            if dist < min_dist:
                min_dist = dist
                best_offset = i
                
        # Expected random distance for 32 bits is 16.
        # < 8 would be significant.
        print(f"Closest Mirror Match in Merkle Root:")
        print(f"Offset: {best_offset}")
        print(f"Distance: {min_dist} bits (Expected ~16)")
        z_score = (16 - min_dist) / math.sqrt(8) # approx
        print(f"Significance: {z_score:.2f} Sigma")
        
        if z_score > 3.0:
            print(">>> SYMMETRY DETECTED: Nonce mirrors Merkle Root segment!")
        else:
            print(">>> RESULT: No hidden reflection found.")

def run_decomposition():
    print("CMFO COMPONENT-WISE FORENSIC DECOMPOSITION")
    analyzer = ComponentAnalyzer()
    
    # 905561 Data
    block = {
        'height': 905561,
        'ver': 598728704,
        'prev': "00000000000000000001c95188c655f79a281d351db7ffad034d39ba3c6be4ce",
        'merkle': "3c38914753b8b54b0fff74ca07e5c998e69523a4f3efa82a871075e46ee233ee",
        'time': 1752527466,
        'bits': 386022054,
        'nonce': 3536931971
    }
    
    analyzer.analyze_block(block)

if __name__ == "__main__":
    run_decomposition()
