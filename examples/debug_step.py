
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bindings', 'python'))

from cmfo.fractal_sha256 import SHA256Fractal, FractalState
from cmfo.fractal_sha256.sha256_engine import SHA256Fractal as Engine

def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

def Ch(x, y, z):
    return (x & y) ^ (~x & z)

def Maj(x, y, z):
    return (x & y) ^ (x & z) ^ (y & z)

def Sigma0(x):
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)

def Sigma1(x):
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)

def debug_full_trace():
    print("\nStarting Full 64-Round Trace...")
    
    # Golden
    H = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    ]
    K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]
    
    W = [0] * 64
    W[0] = 0x80000000
    for t in range(16, 64):
        s1 = rotr(W[t-2], 17) ^ rotr(W[t-2], 19) ^ (W[t-2] >> 10)
        s0 = rotr(W[t-15], 7) ^ rotr(W[t-15], 18) ^ (W[t-15] >> 3)
        W[t] = (W[t-16] + s0 + W[t-7] + s1) & 0xFFFFFFFF
    
    a,b,c,d,e,f,g,h = H
    
    golden_states = []
    
    for t in range(64):
        S1 = Sigma1(e)
        ch = Ch(e,f,g)
        temp1 = (h + S1 + ch + K[t] + W[t]) & 0xFFFFFFFF
        S0 = Sigma0(a)
        maj = Maj(a,b,c)
        temp2 = (S0 + maj) & 0xFFFFFFFF
        
        h = g
        g = f
        f = e
        e = (d + temp1) & 0xFFFFFFFF
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xFFFFFFFF
        golden_states.append([a,b,c,d,e,f,g,h])
        
    # Fractal
    eng = Engine()
    msg = b''
    padded = eng._pad_message(msg)
    block = padded[:64]
    
    for i in range(8):
        eng.state.load_word(i*32, eng.H[i])
        
    # Inject W (Fix for AttributeError)
    W_fractal = [0]*64
    for t in range(16):
        W_fractal[t] = int.from_bytes(block[t*4 : (t+1)*4], 'big')
    for t in range(16, 64):
        s1 = eng._rotr(W_fractal[t-2], 17) ^ eng._rotr(W_fractal[t-2], 19) ^ (W_fractal[t-2] >> 10)
        s0 = eng._rotr(W_fractal[t-15], 7) ^ eng._rotr(W_fractal[t-15], 18) ^ (W_fractal[t-15] >> 3)
        W_fractal[t] = (W_fractal[t-16] + s0 + W_fractal[t-7] + s1) & 0xFFFFFFFF
    eng.W = W_fractal
        
    POS_A_list = [0, 32, 64, 96, 128, 160, 192, 224]
    
    for t in range(64):
        eng._compression_round(t)
        fractal_vals = [eng.state.extract_word(pos) for pos in POS_A_list]
        golden_vals = golden_states[t]
        
        if fractal_vals != golden_vals:
            print(f"Mismatch at Round {t}!")
            names = ['a','b','c','d','e','f','g','h']
            for i in range(8):
                if fractal_vals[i] != golden_vals[i]:
                    print(f"  {names[i]}: Exp {golden_vals[i]:08x}, Got {fractal_vals[i]:08x}")
            return
            
    print("All 64 Rounds MATCH!")
    
    # 3. Verify Final Digest (Single Hash)
    import hashlib
    
    # Debug Final Add
    print("\nDebug Final Add:")
    print("  H0[0]: {:08x}".format(eng.H0[0]))
    # Get Final State a (POS_A)
    final_a = eng.state.extract_word(0)
    print("  Final a: {:08x}".format(final_a))
    
    calc_h0 = (eng.H0[0] + final_a) & 0xFFFFFFFF
    print("  Calc H0: {:08x}".format(calc_h0))
    
    std_hash = hashlib.sha256(b'')
    std_digest = std_hash.digest()
    std_h0 = int.from_bytes(std_digest[:4], 'big')
    print("  Std H0:  {:08x}".format(std_h0))
    
    if calc_h0 == std_h0:
        print("  H0 Sum MATCH!")
    else:
        print("  H0 Sum FAIL!")

    print("\nVerifying Single SHA-256 Digest...")
    # ... execution continues ...
    
    # Use the engine normally (process_block)
    fractal_digest_bytes = eng.hash(b'')
    fractal_digest_hex = fractal_digest_bytes.hex()
    
    std_digest_hex = hashlib.sha256(b'').hexdigest()
    
    print(f"Fractal: {fractal_digest_hex}")
    print(f"Std:     {std_digest_hex}")
    
    if fractal_digest_hex == std_digest_hex:
        print("Single Hash MATCH!")
    else:
        print("Single Hash FAIL mismatch.")

debug_full_trace()
