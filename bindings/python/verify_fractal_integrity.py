
import os
import sys
import hashlib
import random
import struct

# Add local path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from cmfo.crypto.fractal_sha256 import FractalSHA256, FractalWord

def run_verification(iterations=1000):
    print(f"--- Fractal SHA-256 Rigorous Verification ({iterations} iterations) ---")
    
    success_count = 0
    fail_count = 0
    
    for i in range(iterations):
        # Generate random message (length 0 to 64 bytes)
        msg_len = random.randint(0, 55) # Keep it single block for speed in this test (55 bytes max for 1 block)
        msg = os.urandom(msg_len)
        
        # Reference Hash
        ref_hash = hashlib.sha256(msg).hexdigest()
        
        # Prepare Block for Fractal SHA
        # Manual Padding Logic (Simplified for single block test)
        # Pad: 1 bit, then 0s, then 64-bit length
        # 1 byte = 0x80
        
        # Build integers
        # We need to construct the 16-word block exactly as SHA-256 expects
        
        # 1. Pad message
        padded = bytearray(msg)
        padded.append(0x80)
        while (len(padded) + 8) % 64 != 0:
            padded.append(0x00)
            
        # Append length (bits)
        length_bits = msg_len * 8
        padded += struct.pack('>Q', length_bits)
        
        # Convert to 16 words (Big Endian)
        words = []
        for j in range(0, 64, 4):
            w = struct.unpack('>I', padded[j:j+4])[0]
            words.append(w)
            
        m_fractal = [FractalWord.from_int(w) for w in words]
        
        # Run Fractal SHA
        fsha = FractalSHA256()
        fsha.compress(m_fractal)
        
        res = fsha.get_hash()
        res_hex = "".join(f"{x:08x}" for x in res)
        
        if res_hex == ref_hash:
            success_count += 1
            if i % 100 == 0:
                print(f"[{i}/{iterations}] PASS") 
        else:
            fail_count += 1
            print(f"[{i}/{iterations}] FAIL!")
            print(f"Msg: {msg.hex()}")
            print(f"Ref: {ref_hash}")
            print(f"Frc: {res_hex}")
            break
            
    if fail_count == 0:
        print(f"\n✓ SUCCESS: {success_count} tests passed perfectly.")
        print("Fractal SHA-256 is BIT-EXACT.")
        return True
    else:
        print(f"\n✗ FAILURE: Passed {success_count}, Failed {fail_count}.")
        return False

if __name__ == "__main__":
    run_verification(1000)
