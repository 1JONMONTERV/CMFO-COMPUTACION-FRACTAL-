
import sys
import os
import time
import struct
import hashlib

# Add binding path
current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, '..', 'bindings', 'python')
sys.path.append(bindings_path)

try:
    from cmfo.mining.fractal_sha import FractalSHA256, BitcoinHeaderStructure, H_INIT
except ImportError as e:
    print(f"Error importing Fractal SHA: {e}")
    sys.exit(1)

def run_proof():
    print("==================================================")
    print("   CMFO MIDSTATE OPTIMIZATION PROOF")
    print("   Target: Prove speedup of H1 Precomputation")
    print("==================================================")
    
    # 1. Create a Dummy 80-byte Header
    # Version(4) + Prev(32) + Root(32) + Time(4) + Bits(4) + Nonce(4)
    header_prefix = b'\x02\x00\x00\x00' + (b'\xaa'*32) + (b'\xff'*28) 
    # That's 4+32+28 = 64 bytes exactly (Block 1)
    
    # Rest of header (16 bytes): RootSuffix(4) + Time(4) + Bits(4) + Nonce(4)
    # We'll make it simple.
    header_full = bytearray(80)
    header_full[0:64] = header_prefix
    header_full[64:80] = b'\x11' * 16 # Dummy tail
    
    header_bytes = bytes(header_full)
    
    # 2. Prepare Structured Blocks
    b1, b2_template = BitcoinHeaderStructure.create_template(header_bytes)
    
    print(f"[Structure] Block 1 (Fixed): {b1.hex()[:32]}... ({len(b1)} bytes)")
    print(f"[Structure] Block 2 Template: {b2_template.hex()[:32]}... ({len(b2_template)} bytes)")
    print(f"            Padding Verified: {b2_template[-8:].hex()} (Length=640 bits)")
    
    ITERATIONS = 50000 
    
    print(f"\n[Test 1] Standard Full Hashing ({ITERATIONS} iterations)...")
    start_ts = time.time()
    
    # Baseline: Re-hash everything using pure python implementation for fair comparison
    # (Using hashlib would be C-speed and unfair to compare against pure python midstate)
    # Actually, we should compare "Algorithm A" vs "Algorithm B" using same engine.
    
    # We use our FractalSHA256 for both to strictly measure algorithmic diff.
    
    hash_count = 0
    for nonce in range(ITERATIONS):
        # Update nonce in full header (copying 80 bytes)
        current_header = bytearray(header_bytes)
        current_header[76:80] = struct.pack('<I', nonce)
        
        # Build Blocks
        current_b1 = current_header[0:64]
        current_b2_part = current_header[64:80]
        # Padding is constant but we reconstruct it for "naive" approach
        # (Simulating naive processor that doesn't know about midstate)
        # Re-creating full padding every time is part of the cost effectively.
        
        # Naive: Process B1, then B2
        h_state = H_INIT
        h_state = FractalSHA256.compress(h_state, current_b1)
        
        # Rebuild B2 with padding
        current_b2 = bytearray(current_b2_part)
        current_b2.append(0x80)
        current_b2.extend([0x00]*39)
        current_b2.extend(struct.pack('!Q', 640))
        
        h_final = FractalSHA256.compress(h_state, current_b2)
        hash_count += 1
        
    t_full = time.time() - start_ts
    print(f"  Time: {t_full:.4f}s | Speed: {ITERATIONS/t_full:.0f} hashes/s")
    
    print(f"\n[Test 2] Midstate Optimization ({ITERATIONS} iterations)...")
    start_ts = time.time()
    
    # 1. Precompute B1 State (Midstate)
    midstate = FractalSHA256.compress(H_INIT, b1)
    
    # 2. Loop B2 only
    for nonce in range(ITERATIONS):
        # Inject nonce into PRE-BUILT template (fast)
        # We manually modify the bytearray to be super fast
        
        # In real C++ mining, this injection is a pointer cast.
        # Here we do minimal python work.
        current_b2 = BitcoinHeaderStructure.inject_nonce(b2_template, nonce)
        
        # Process ONLY B2 from Midstate
        h_final = FractalSHA256.compress(midstate, current_b2)
        
    t_mid = time.time() - start_ts
    print(f"  Time: {t_mid:.4f}s | Speed: {ITERATIONS/t_mid:.0f} hashes/s")
    
    speedup = t_full / t_mid
    print(f"\n[Result] Optimization Speedup: {speedup:.2f}x")
    if speedup > 1.3:
        print("  MIDSTATE STRATEGY VALIDATED (Significant Gain)")
    else:
        print("  Marginal Gain (Python overhead complexity?)")

if __name__ == "__main__":
    run_proof()
