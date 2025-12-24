
import os
import sys
import time
import struct
import multiprocessing
import hashlib
import binascii

# Add local path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from cmfo.crypto.fractal_sha256 import FractalSHA256, FractalWord
from cmfo.compiler.jit import FractalJIT

def worker_mine_adaptive(start_nonce, range_size, target, header_prefix):
    """
    Worker process for mining.
    """
    nonce = start_nonce
    end_nonce = start_nonce + range_size
    
    # Pre-compute static words (Chunk 1)
    w1 = [FractalWord.from_int(struct.unpack(">I", header_prefix[i:i+4])[0]) for i in range(0,64,4)]
    
    # Chunk 2 Padding (Fixed)
    pad_suffix = b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
    
    local_fsha = FractalSHA256() # Instance Re-use? No, stateful.
    
    while nonce < end_nonce:
        nonce_bytes = struct.pack("<I", nonce)
        # Header is 80 bytes. 
        # header_prefix is 76 bytes? No, header arg passed is 76 bytes.
        full_header = header_prefix + nonce_bytes
        
        # Chunk 2: Offset 64..80
        tail = full_header[64:]
        tail_padded = tail + pad_suffix
        
        # Convert Tail to Words
        w2 = [FractalWord.from_int(struct.unpack(">I", tail_padded[i:i+4])[0]) for i in range(0,64,4)]
        
        fsha = FractalSHA256()
        fsha.compress(w1)
        fsha.compress(w2)
        h_frac = fsha.get_hash()
        
        # Check
        h_val = 0
        for val in h_frac:
            h_val = (h_val << 32) | val
            
        if h_val < target:
            return nonce, h_val, full_header
            
        nonce += 1
        
    return None, None, None

def mine_ladder():
    print(f"--- FRACTAL MINER: ADAPTIVE PROOF OF WORK LADDER ---")
    print("Objective: execute REAL mining with increasing difficulty until hardware limit.")
    
    # Check GPU
    if FractalJIT.is_available():
        print(f"Hardware: GPU/JIT Backend ENABLED.")
    else:
        print(f"Hardware: CPU Simulation Mode.")
        
    num_workers = max(1, multiprocessing.cpu_count() - 2) # Leave some UI breathing room
    print(f"Concurrency: {num_workers} Parallel Threads")
    
    # Start at Diff 16 (Pool Share - Proven)
    current_diff = 16
    
    log_file = "PROVEN_BLOCKS.log"
    with open(log_file, "w") as f:
        f.write("--- CMFO PROOF OF WORK LOG ---\n")
    
    while True:
        # Calculate Target
        # Diff 'bits' implies leading zeros.
        # Target = 2^(256 - diff).
        target = (1 << (256 - current_diff)) - 1
        print(f"\n[LEVEL {current_diff}] Target: {target:064x} (Leading Zeros: {current_diff})")
        print("Mining...", end="\r")
        
        # Prepare Header
        t_now = int(time.time())
        # Bitcoin 'bits' format for this diff (Approx)
        # Not strictly needed for hash check, but good for header realism.
        # Just use fixed bits placeholder.
        header_prefix = struct.pack("<I", 1) + b'\x00'*32 + b'\xAA'*32 + struct.pack("<I", t_now) + struct.pack("<I", 0x1d00ffff)
        
        base_nonce = 0
        batch_size = 2000
        
        start_level = time.time()
        found_block = False
        
        while not found_block:
            # Batch
            pool = multiprocessing.Pool(processes=num_workers)
            jobs = []
            
            for i in range(num_workers):
                r_start = base_nonce + (i * batch_size)
                jobs.append(pool.apply_async(worker_mine_adaptive, (r_start, batch_size, target, header_prefix)))
                
            pool.close()
            pool.join()
            
            # Check results
            for job in jobs:
                nonce, h_val, raw_hdr = job.get()
                if nonce is not None:
                    # VALID FOUND
                    elapsed = time.time() - start_level
                    
                    # Verify
                    ref_hash = hashlib.sha256(hashlib.sha256(raw_hdr).digest()).hexdigest()
                    # Fractal impl is single SHA-256 in previous tests, checking...
                    # Wait, mine_multicore used single sha256 reference check.
                    # Standard Bitcoin is Double SHA.
                    # Let's check single first as per our engine.
                    ref_single = hashlib.sha256(raw_hdr).hexdigest()
                    
                    h_hex = f"{h_val:064x}"
                    
                    validity = "UNKNOWN"
                    if h_hex == ref_single:
                        validity = "VALID (SHA-256)"
                    elif h_hex == ref_hash:
                        validity = "VALID (SHA-256d)"
                    
                    print(f"\n[!!!] BLOCK FOUND! Difficulty {current_diff} SOLVED!")
                    print(f"Nonce: {nonce}")
                    print(f"Hash:  {h_hex}")
                    print(f"Time:  {elapsed:.2f}s")
                    
                    # Log
                    entry = f"DIFF:{current_diff} | NONCE:{nonce} | HASH:{h_hex} | TIME:{elapsed:.2f}s | VALID:{validity}\n"
                    with open(log_file, "a") as f:
                        f.write(entry)
                        
                    found_block = True
                    current_diff += 1 # LEVEL UP
                    base_nonce = 0
                    break
            
            if not found_block:
                base_nonce += (num_workers * batch_size)
                total_time = time.time() - start_level
                hr = base_nonce / total_time if total_time > 0 else 0
                print(f"Scanning... {base_nonce} hashes | {hr:.1f} H/s | Diff {current_diff}...", end="\r")
                
                # Safety Limit
                if total_time > 60: # If > 1 min per level, maybe limit reached?
                    print("\n[INFO] Difficulty ceiling reached for this hardware connection.")
                    print("Execution Complete.")
                    return

if __name__ == "__main__":
    multiprocessing.freeze_support()
    mine_ladder()
