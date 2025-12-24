"""
Mining Sampler for Topology Analysis
====================================

Generates a stratified dataset of block headers with varying difficulty.
Target:
- 100 samples with 0 leading zeros (Random)
- 100 samples with ~8 leading zeros (Easy)
- 100 samples with ~12 leading zeros (Medium)
- 100 samples with ~16 leading zeros (Hard)

Uses standard hashlib for speed to generate the dataset.
The analysis will then use CMFO Fractal Algebra.
"""

import hashlib
import json
import os
import time
import struct
import random

def sha256d(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def count_leading_zeros(h):
    # h is bytes
    zeros = 0
    for b in h:
        if b == 0:
            zeros += 8
        else:
            # Count zero bits in b
            # 0000 0001 -> 7 zeros?
            # SHA256 target is usually Big Endian comparison or Little Endian?
            # Bitcoin treats hash as Little Endian number for difficulty check,
            # but usually displayed as Big Endian string with leading zeros.
            # We will count leading zeros in the Big Endian byte representation (standard display).
            # e.g. 000000..
            zeros += bin(b)[2:].zfill(8).find('1')
            break
    return zeros

def mine_samples(target_zeros, count=100, timeout=10.0):
    samples = []
    
    # Dummy block header components
    ver = struct.pack("<I", 1)
    prev = b'\x00' * 32
    root = b'\x00' * 32 # Varies?
    ts = struct.pack("<I", int(time.time()))
    bits = b'\xff\xff\xff\xff' # Easy target
    
    # We vary Root and Nonce
    
    start_time = time.time()
    nonce = 0
    
    print(f"Mining for {target_zeros} zeros (Target: {count})...")
    
    while len(samples) < count:
        if time.time() - start_time > timeout:
            print(f"  Timeout reached. Found {len(samples)}.")
            break
            
        # Varyng root slightly to avoid nonce exhaustion in fixed space
        if nonce > 0xFFFFFFFF:
            nonce = 0
            # Change root
            root = os.urandom(32)
            
        nonce_bytes = struct.pack("<I", nonce)
        header = ver + prev + root + ts + bits + nonce_bytes
        
        h = sha256d(header)
        # Reverse hash for BE display check
        h_be = h[::-1]
        
        z = count_leading_zeros(h_be)
        
        # We want EXACTLY target_zeros or AT LEAST?
        # For analysis, "At least" is better, but stratification implies "Around".
        # Let's verify "At least".
        # Actually to see gradients, we want bins.
        # Let's store ANYTHING that matches our requested bins.
        
        if z >= target_zeros:
            # If we want exact bins, we skip if z > target_zeros + 2 maybe?
            # Let's just collect >= for the 'Hard' bin, and exact for low bins.
            if target_zeros == 0:
                 # Random. most have 0.
                 pass 
            
            samples.append({
                "header_hex": header.hex(),
                "hash_hex": h_be.hex(),
                "zeros": z
            })
            
        nonce += 1
        
    return samples

def main():
    dataset = {}
    
    # Bin 0 (Randomish) - accept anything, but statistically majority is 0-3 zeros.
    dataset['diff_0'] = mine_samples(0, count=100)
    
    # Bin 8
    dataset['diff_8'] = mine_samples(8, count=100, timeout=15)
    
    # Bin 12
    dataset['diff_12'] = mine_samples(12, count=100, timeout=20)
    
    # Bin 16 (might take a moment)
    dataset['diff_16'] = mine_samples(16, count=100, timeout=30)
    
    out_path = os.path.join(os.path.dirname(__file__), 'mining_dataset.json')
    with open(out_path, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Dataset saved to {out_path}")
    print(f"Counts: 0={len(dataset['diff_0'])}, 8={len(dataset['diff_8'])}, 12={len(dataset['diff_12'])}, 16={len(dataset['diff_16'])}")

if __name__ == "__main__":
    main()
