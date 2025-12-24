
import sys
import os
import math

# Add local path to import cmfo
try:
    from ..logic.phi_logic import fractal_and, fractal_or, fractal_xor, phi_sign
except ImportError:
    # Fallback
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from cmfo.logic.phi_logic import fractal_and, fractal_or, fractal_xor, phi_sign


# Constants
WORD_BITS = 32
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

IV = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

class FractalWord:
    """
    A word of N trits (floats in [-1, 1]).
    Represents a generalized register for Fractal SHA-256.
    """
    def __init__(self, trits=None, size=32):
        self.size = size
        if trits is None:
            self.trits = [0.0] * size
        else:
            if len(trits) != size:
                # Pad or truncate? Strict for now.
                if len(trits) < size:
                     self.trits = trits + [0.0]*(size - len(trits))
                else:
                     self.trits = trits[:size]
            else:
                self.trits = list(trits)


    @classmethod
    def from_int(cls, value, size=32):
        """Standard binary to fractal word (0.0, 1.0)."""
        trits = []
        for i in range(size):
            # [MSB ... LSB]
            bit = (value >> (size - 1 - i)) & 1
            trits.append(1.0 if bit else 0.0)
        return cls(trits, size)

    def to_int(self):
        """Collapse to standard int."""
        val = 0
        for i, t in enumerate(self.trits):
            bit = 1 if t >= 0.5 else 0
            val |= (bit << (self.size - 1 - i))
        return val

    # --- Fractal Logic Gates (Local Definitions for SHA consistency) ---
    
    def __xor__(self, other):
        # Fractal XOR: Absolute difference preserves "distance"
        # 1^1 = 0, 0^0 = 0, 1^0 = 1.
        # Fractal: |1.0 - 0.9| = 0.1 (Small Diff).
        return FractalWord([abs(a - b) for a, b in zip(self.trits, other.trits)], self.size)
    
    def __and__(self, other):
        # Fractal AND: Product
        # 1&1=1, others 0.
        # Fractal: x*y.
        return FractalWord([a * b for a, b in zip(self.trits, other.trits)], self.size)
    
    def __or__(self, other):
        # Fractal OR: x + y - x*y
        # 1|1=1.
        return FractalWord([a + b - (a * b) for a, b in zip(self.trits, other.trits)], self.size)
    
    def __invert__(self):
        # NOT: 1 - x
        return FractalWord([1.0 - x for x in self.trits], self.size)

    # --- Shift / Rotate ---
    
    def rotr(self, n):
        """Rotate Right"""
        n = n % self.size
        new_trits = self.trits[-n:] + self.trits[:-n]
        return FractalWord(new_trits, self.size)

    def shr(self, n):
        """Shift Right (fill with 0.0)"""
        if n >= self.size:
             return FractalWord([0.0]*self.size, self.size)
        new_trits = [0.0]*n + self.trits[:-n]
        return FractalWord(new_trits, self.size)

    # --- Arithmetic ---

    def add(self, other):
        """
        Fractal Ripple Carry Adder.
        """
        sum_trits = [0.0] * self.size
        cin = 0.0 
        
        for i in reversed(range(self.size)):
            a = self.trits[i]
            b = other.trits[i]
            
            # XOR: abs(a-b)
            # AND: a*b
            # OR: a+b - ab
            
            # Sum = a ^ b ^ cin
            axb = abs(a - b)
            s = abs(axb - cin)
            
            # Cout = (a&b) | (cin & (a^b))
            #      = ab + cin*axb - ab*cin*axb
            ab = a * b
            cin_axb = cin * axb
            
            # Logic OR
            cout = ab + cin_axb - (ab * cin_axb)
            
            sum_trits[i] = s
            cin = cout
            
        return FractalWord(sum_trits, self.size)

    def __repr__(self):
        val = self.to_int()
        # Avg deviation from closest int
        dev = sum(min(t, abs(1-t)) for t in self.trits)/self.size
        return f"FractalWord({hex(val)}, ~{dev:.3f})"

# --- SHA-256 Functions ---

def f_sigma0(x): return x.rotr(7) ^ x.rotr(18) ^ x.shr(3)
def f_sigma1(x): return x.rotr(17) ^ x.rotr(19) ^ x.shr(10)
def f_Sigma0(x): return x.rotr(2) ^ x.rotr(13) ^ x.rotr(22)
def f_Sigma1(x): return x.rotr(6) ^ x.rotr(11) ^ x.rotr(25)

def f_Ch(x, y, z):
    # (x AND y) XOR ((NOT x) AND z)
    # Optimized: (x & y) ^ (~x & z)
    return (x & y) ^ ((~x) & z)

def f_Maj(x, y, z):
    # (x AND y) XOR (x AND z) XOR (y AND z)
    # Using Fractal gates
    t1 = x & y
    t2 = x & z
    t3 = y & z
    return t1 ^ t2 ^ t3


class FractalSHA256:
    def __init__(self):
        self.H = [FractalWord.from_int(iv) for iv in IV]
        self.W_fractal = []

    def prepare_schedule(self, message_block):
        """
        message_block: List of 16 FractalWords
        """
        self.W_fractal = message_block[:]
        for t in range(16, 64):
            # W[t] = Sigma1(W[t-2]) + W[t-7] + Sigma0(W[t-15]) + W[t-16]
            # Use Fractal Operations
            wt2 = self.W_fractal[t-2]
            wt7 = self.W_fractal[t-7]
            wt15 = self.W_fractal[t-15]
            wt16 = self.W_fractal[t-16]
            
            s1 = f_sigma1(wt2)
            s0 = f_sigma0(wt15)
            
            # Additions
            step1 = s1.add(wt7)
            step2 = step1.add(s0)
            step3 = step2.add(wt16)
            
            self.W_fractal.append(step3)

    def compress(self, message_block):
        """
        Run 64 rounds of compression on a message block.
        """
        self.prepare_schedule(message_block)
        
        a, b, c, d = self.H[0], self.H[1], self.H[2], self.H[3]
        e, f, g, h = self.H[4], self.H[5], self.H[6], self.H[7]
        
        for t in range(64):
            kt = FractalWord.from_int(K[t])
            wt = self.W_fractal[t]
            
            # T1 = h + Sigma1(e) + Ch(e,f,g) + Kt + Wt
            s1 = f_Sigma1(e)
            ch = f_Ch(e, f, g)
            
            t1 = h.add(s1).add(ch).add(kt).add(wt)
            
            # T2 = Sigma0(a) + Maj(a,b,c)
            s0 = f_Sigma0(a)
            maj = f_Maj(a, b, c)
            
            t2 = s0.add(maj)
            
            # Update state
            h = g
            g = f
            f = e
            e = d.add(t1)
            d = c
            c = b
            b = a
            a = t1.add(t2)
            
        # Update H (Feedforward)
        self.H[0] = self.H[0].add(a)
        self.H[1] = self.H[1].add(b)
        self.H[2] = self.H[2].add(c)
        self.H[3] = self.H[3].add(d)
        self.H[4] = self.H[4].add(e)
        self.H[5] = self.H[5].add(f)
        self.H[6] = self.H[6].add(g)
        self.H[7] = self.H[7].add(h)
        
        return self.H

    def get_hash(self):
        """Return bits."""
        return [h.to_int() for h in self.H]

if __name__ == "__main__":
    import struct
    
    print("--- Fractal SHA-256 Full Test ---")
    
    # Create a message block: "abc" padded
    # 'abc' = 61 62 63 80 ...
    # 0x61626380
    m_int = [0] * 16
    m_int[0] = 0x61626380
    m_int[15] = 24 # 3 bytes * 8 bits
    
    # Convert to FractalWords
    m_fractal = [FractalWord.from_int(x) for x in m_int]
    
    # Standard Hashing for reference
    import hashlib
    ref_hash = hashlib.sha256(b"abc").hexdigest()
    print(f"Reference: {ref_hash}")
    
    # Fractal Hashing
    fsha = FractalSHA256()
    fsha.compress(m_fractal)
    
    res = fsha.get_hash()
    res_hex = "".join(f"{x:08x}" for x in res)
    print(f"Fractal  : {res_hex}")
    
    if res_hex == ref_hash:
        print("✓ SUCCESS: Fractal SHA-256 matches Standard.")
    else:
        print("✗ FAILURE: Mismatch.")

    print("\n--- Reversibility/Traceability Demo ---")
    print("Propagating a small perturbation (0.01) in input bit 0...")
    
    # Perturbed input
    m_fuzzy = [FractalWord.from_int(x) for x in m_int]
    # Add perturbation to bit 0 of word 0
    # Original is 0.0 (from 0x...80, LSB is 0). Make it 0.01.
    m_fuzzy[0].trits[31] = 0.01 
    
    fsha_fuzzy = FractalSHA256()
    fsha_fuzzy.compress(m_fuzzy)
    
    # Check output deviation
    print("Checking H[0] deviation:")
    h0 = fsha_fuzzy.H[0]
    devs = [t - (1.0 if t >= 0.5 else 0.0) for t in h0.trits]
    print(f"H[0] value: {hex(h0.to_int())}")
    print(f"Deviations (first 8 trits): {[f'{d:.4f}' for d in devs[:8]]}")
    
    has_trace = any(abs(d) > 1e-9 for d in devs)
    if has_trace:
        print("✓ Trace Detected: The 0.01 perturbation propagated to the hash signature!")
    else:
        print("✗ Trace Lost: The perturbation vanished.")
