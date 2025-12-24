
import os
import sys
import struct
import random
import binascii

# Add root path to find cmfo_inverse_solver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import Solver
try:
    from cmfo_inverse_solver import InverseGeometricSolver
except ImportError:
    # Fallback if in different location or PYTHONPATH issue
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
    from cmfo_inverse_solver import InverseGeometricSolver

def reverse_hex(h):
    return binascii.unhexlify(h)[::-1]

def prove_geometric_resonance():
    print("--- REAL BLOCK DATA VERIFICATION (FRACTAL GEOMETRY) ---")
    print("Target: Block 905561 (From CSV)")
    
    # Data from API/CSV
    ver = 598728704
    # Note: Explorer hashes are Big Endian display, internal Header is Little Endian.
    # So we reverse them.
    prev_hex = "00000000000000000001c95188c655f79a281d351db7ffad034d39ba3c6be4ce"
    merkle_hex = "3c38914753b8b54b0fff74ca07e5c998e69523a4f3efa82a871075e46ee233ee"
    
    prev = reverse_hex(prev_hex)
    merkle = reverse_hex(merkle_hex)
    
    time_val = 1752527466
    bits = 386022054
    real_nonce = 3536931971
    
    header_tmpl = struct.pack("<I", ver) + prev + merkle + struct.pack("<I", time_val) + struct.pack("<I", bits) + b'\x00\x00\x00\x00'
    
    print(f"Header Configured.")
    print(f"Real Nonce: {real_nonce}")
    
    print(f"Header Configured.")
    print(f"Real Nonce: {real_nonce}")
    
    # Use FractalSHA256 directly to measure Resonance (Zeros)
    from cmfo.crypto.fractal_sha256 import FractalSHA256, FractalWord
    
    def get_fractal_resonance(header, nonce):
        # Construct header with nonce
        h = bytearray(header)
        h[76:80] = struct.pack("<I", nonce)
        h_bytes = bytes(h)
        
        # Prepare for FractalSHA
        # Pad standard 80 bytes -> 128 bytes
        # 80 bytes + 0x80 + 0x00... + len(640)
        # 640 bits = 80 bytes * 8
        padded = h_bytes + b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
        
        # Split into 2 chunks
        chunk1 = padded[:64]
        chunk2 = padded[64:]
        
        w1 = [FractalWord.from_int(struct.unpack(">I", chunk1[i:i+4])[0]) for i in range(0,64,4)]
        w2 = [FractalWord.from_int(struct.unpack(">I", chunk2[i:i+4])[0]) for i in range(0,64,4)]
        
        fsha = FractalSHA256()
        fsha.compress(w1)
        fsha.compress(w2)
        h1_ints = fsha.get_hash()
        
        # SHA-256d: Hash the Hash
        # Convert H1 ints to bytes (Big Endian standard output of SHA)
        h1_bytes = b"".join(struct.pack(">I", x) for x in h1_ints)
        
        # Prepare H2 Input (32 bytes)
        # Pad: 32 + 1 (x80) + 23 (00) + 8 (len=256) = 64 bytes
        pad2 = h1_bytes + b'\x80' + b'\x00'*23 + struct.pack(">Q", 256)
        
        w_final = [FractalWord.from_int(struct.unpack(">I", pad2[i:i+4])[0]) for i in range(0,64,4)]
        
        fsha2 = FractalSHA256()
        fsha2.compress(w_final)
        h2_ints = fsha2.get_hash()
        
        # Bitcoin uses Little Endian display for the final hash,
        # but the zeros check is on the numeric value (Big Endian interpretation of the byte array? 
        # No, Target comparison is usually done on the 256-bit number).
        # However, checking Leading Zeros on the standard hash output (h2_ints) is correct.
        
        # Bitcoin Difficulty Check:
        # Internal SHA256d is Little Endian.
        # But Target is checked against Big Endian interpretation.
        # So we must reverse the bytes of the full hash to see "Leading Zeros".
        
        # Convert ints to bytes (Little Endian words? No, usually SHA output is BE words).
        # Let's assume h2_ints follows standard [H0..H7].
        # If it printed 'fd4a...', then H0 is fd4a...
        # So we currently have 'fd...00'.
        # We want '00...fd'.
        
        # Construct full byte array
        final_bytes = b"".join(struct.pack(">I", x) for x in h2_ints)
        
        # Reverse for BE display/check
        be_bytes = final_bytes[::-1]
        
        # Convert to hex for verification
        be_hex = be_bytes.hex()
        
        # Count leading zeros on be_bytes
        zeros = 0
        for b in be_bytes:
            if b == 0:
                zeros += 8
            else:
                # Count zero bits in byte
                lz = 0
                for k in range(8):
                    if (b >> (7-k)) & 1:
                        break
                    lz += 1
                zeros += lz
                break
                
        # Return hash as BE ints for printing
        be_ints = [struct.unpack(">I", be_bytes[i:i+4])[0] for i in range(0,32,4)]
        return zeros, be_ints

    # 1. Analyze Real Nonce
    res_real, hash_real = get_fractal_resonance(header_tmpl, real_nonce)
    print(f"\n[ANALYSIS] Real Nonce Resonance (Zero Depth): {res_real} bits")
    print(f"Hash: {''.join(f'{x:08x}' for x in hash_real)}")
    
    if res_real > 60:
         print(">> STATUS: HYPER-RESONANT ATTRACTOR CONFIRMED.")
    else:
         print(f">> STATUS: Weak Resonance ({res_real}). Strange for Mainnet.")

    # 2. Compare against Random Noise
    print("\nComparing against 100 Random Candidates...")
    random_res = []
    
    for _ in range(100):
        r = random.randint(0, 2**32-1)
        res, _ = get_fractal_resonance(header_tmpl, r)
        random_res.append(res)
            
    avg_rand = sum(random_res)/len(random_res)
    max_rand = max(random_res)
    
    print(f"Random Field Avg Resonance: {avg_rand:.2f}")
    print(f"Random Field Max Resonance: {max_rand}")
    
    print("\n" + "="*50)
    print("VERIFICATION RESULT")
    print("="*50)
    
    if res_real > max_rand:
        factor = res_real / (avg_rand if avg_rand > 0 else 1)
        print(f"The Real Nonce is {factor:.1f}x more resonant than noise.")
        print("CONCLUSION: Validated. The Nonce sits in a deep fractal well.")
    else:
        print("CONCLUSION: Failed to distinguish.")

if __name__ == "__main__":
    prove_geometric_resonance()
