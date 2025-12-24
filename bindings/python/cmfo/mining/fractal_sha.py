"""
CMFO Mining - Fractal SHA-256 (Midstate & Structure)
====================================================

Implements:
1. Exact Bitcoin Block Header Structure (80 bytes + Padding = 128 bytes).
2. Merkle-Damgard Chaining (Midstate Optimization).
3. Pure Python implementation of SHA-256 compression for transparency.

Structure enforced:
  [ Block 1 (64 bytes) ]  ||  [ Block 2 (64 bytes) ]
  Header[0..63]               Header[64..79] + Padding
"""

import struct
import copy

# SHA-256 Constants (K)
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

# Initial Hash Values (H0)
H_INIT = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

def rotr(x, n): return ((x >> n) | (x << (32 - n))) & 0xffffffff
def sigma0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
def sigma1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
def ch(x, y, z): return (x & y) ^ (~x & z)
def maj(x, y, z): return (x & y) ^ (x & z) ^ (y & z)
def big_sigma0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def big_sigma1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)

class FractalSHA256:
    """
    Stateful SHA-256 Engine allowing Midstate access.
    """
    
    @staticmethod
    def compress(state, block_bytes):
        """
        Processes a 512-bit (64-byte) block and updates the state.
        state: list of 8 integers.
        block_bytes: 64 bytes.
        """
        # Prepare message schedule W
        W = [0] * 64
        # First 16 words from block (Big Endian)
        W[0:16] = struct.unpack('!16L', block_bytes)
        
        for i in range(16, 64):
            W[i] = (sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]) & 0xffffffff
            
        a, b, c, d, e, f, g, h_val = state
        
        for i in range(64):
            t1 = (h_val + big_sigma1(e) + ch(e, f, g) + K[i] + W[i]) & 0xffffffff
            t2 = (big_sigma0(a) + maj(a, b, c)) & 0xffffffff
            
            h_val = g
            g = f
            f = e
            e = (d + t1) & 0xffffffff
            d = c
            c = b
            b = a
            a = (t1 + t2) & 0xffffffff
            
        # Add compressed chunk to current state
        new_state = [
            (state[0] + a) & 0xffffffff,
            (state[1] + b) & 0xffffffff,
            (state[2] + c) & 0xffffffff,
            (state[3] + d) & 0xffffffff,
            (state[4] + e) & 0xffffffff,
            (state[5] + f) & 0xffffffff,
            (state[6] + g) & 0xffffffff,
            (state[7] + h_val) & 0xffffffff
        ]
        return new_state

class BitcoinHeaderStructure:
    """
    Manages the 1024-bit structure for Bitcoin Headers.
    """
    @staticmethod
    def create_template(header_80_bytes):
        """
        Returns (Block1, Block2_Template).
        Block1 is fixed for the header prefix.
        Block2_Template includes the padding, missing only the Nonce update.
        """
        if len(header_80_bytes) != 80:
            raise ValueError("Bitcoin Header must be 80 bytes")
            
        # 1. Split Header
        part1 = header_80_bytes[0:64]    # 64 bytes -> Block 1
        part2 = header_80_bytes[64:80]   # 16 bytes -> Start of Block 2
        
        # 2. Build Padding for Block 2
        # Standard SHA-256 Padding for 80 byte message:
        # - Append '1' bit (0x80 byte)
        # - Append Zeros until length = 448 mod 512 (bytes 56)
        # - Append Length (64-bit Big Endian)
        
        padding = bytearray()
        padding.append(0x80)                 # Byte 80 (relative to total) -> Byte 16 of B2
        padding.extend([0x00] * 39)          # Bytes 81..119 -> Bytes 17..55 of B2
        
        # Length in bits = 80 * 8 = 640
        # 64-bit BE: 0x0000000000000280
        length_bits = 80 * 8
        padding.extend(struct.pack('!Q', length_bits)) # Bytes 120..127 -> Bytes 56..63 of B2
        
        block1 = part1
        block2_template = part2 + padding
        
        assert len(block1) == 64
        assert len(block2_template) == 64
        
        return block1, block2_template

    @staticmethod
    def inject_nonce(block2_template, nonce_val):
        """
        Updates the Nonce (bytes 12-15 of Block 2, i.e., bytes 76-79 of header).
        Nonce is Little Endian in Bitcoin header, but strictly speaking 
        we just put bytes there.
        """
        # Block2 layout:
        # 0..11: Rest of header (Time, Bits usually here)
        # 12..15: Nonce
        # 16..63: Padding
        
        b2 = bytearray(block2_template)
        # Inject 32-bit nonce (Little Endian standard for BTC)
        b2[12:16] = struct.pack('<I', nonce_val) 
        return bytes(b2)
