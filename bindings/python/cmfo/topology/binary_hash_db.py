"""
CMFO - Binary Hash Database (Rigorous Implementation)
======================================================

Implements the exact binary layout specification:
- File 1: hashes_by_i.bin (direct index lookup)
- File 2: prefix_index.bin (bucket metadata)
- File 3: prefix_lists.bin (bucket contents)

Layout follows specification with:
- 128-byte header (little-endian)
- Cryptographic integrity (SHA-256 checksums)
- O(1) index lookup, O(count) prefix lookup
"""

import struct
import hashlib
import os
from .procedural_512 import ProceduralSpace512

class BinaryHashDB:
    """
    Binary hash database with rigorous layout and verification.
    """
    
    # Magic numbers
    MAGIC_HASHES = b'CMFOHSH1'
    MAGIC_PREFIX = b'CMFOPFX1'
    
    # Constants
    VERSION = 1
    HASH_ID = 1  # SHA256d
    MSG_LEN = 64  # 64-byte messages
    HASH_LEN = 32  # 32-byte hashes
    HEADER_SIZE = 128
    
    def __init__(self, base_path='hash_db'):
        self.base_path = base_path
        self.file_hashes = f"{base_path}_hashes_by_i.bin"
        self.file_prefix_idx = f"{base_path}_prefix_index.bin"
        self.file_prefix_lists = f"{base_path}_prefix_lists.bin"
        
        self.space = ProceduralSpace512()
        self.N = 0
        self.prefix_bytes = 2  # k=2 -> 65536 buckets
        self.buckets = 256 ** self.prefix_bytes
    
    def sha256d(self, message):
        """SHA256d: SHA256(SHA256(m))"""
        h1 = hashlib.sha256(message).digest()
        h2 = hashlib.sha256(h1).digest()
        return h2
    
    def write_header_hashes(self, f, N, seed=0):
        """
        Writes 128-byte header for hashes_by_i.bin
        """
        header = bytearray(self.HEADER_SIZE)
        
        # Magic (8 bytes)
        header[0:8] = self.MAGIC_HASHES
        
        # Version (4 bytes LE)
        struct.pack_into('<I', header, 8, self.VERSION)
        
        # Hash ID (4 bytes LE)
        struct.pack_into('<I', header, 12, self.HASH_ID)
        
        # Message length (4 bytes LE)
        struct.pack_into('<I', header, 16, self.MSG_LEN)
        
        # Reserved (4 bytes)
        struct.pack_into('<I', header, 20, 0)
        
        # N - number of entries (8 bytes LE)
        struct.pack_into('<Q', header, 24, N)
        
        # base_i (8 bytes LE)
        struct.pack_into('<Q', header, 32, 0)
        
        # Seed (16 bytes for 128-bit seed)
        struct.pack_into('<Q', header, 40, (seed >> 64) & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into('<Q', header, 48, seed & 0xFFFFFFFFFFFFFFFF)
        
        # Seed length (4 bytes LE)
        struct.pack_into('<I', header, 56, 16)
        
        # Generator ID (4 bytes LE)
        struct.pack_into('<I', header, 60, 1)
        
        # Seed bytes (32 bytes) - store seed as bytes
        seed_bytes = seed.to_bytes(16, 'little')
        header[64:80] = seed_bytes
        
        # payload_sha256 (32 bytes) - placeholder, will update later
        # header[96:128] remains zeros for now
        
        f.write(header)
    
    def generate_hashes(self, N, seed=0):
        """
        Step 1: Generate N hashes and write to hashes_by_i.bin
        """
        print(f"\n[Generation] Creating {N:,} hash entries...")
        print(f"  Output: {self.file_hashes}")
        
        with open(self.file_hashes, 'wb') as f:
            # Write header (payload_sha256 placeholder)
            self.write_header_hashes(f, N, seed)
            
            # Generate and write hashes
            payload_hasher = hashlib.sha256()
            
            for i in range(N):
                # Generate message
                x = i
                y = i * 2
                message = self.space.coords_to_block(x, y)
                
                # Compute hash
                hash_bytes = self.sha256d(message)
                
                # Write to file
                f.write(hash_bytes)
                
                # Update payload hash
                payload_hasher.update(hash_bytes)
                
                if (i + 1) % 10000 == 0:
                    print(f"  Progress: {i+1:,}/{N:,}", end='\r')
            
            # Compute payload integrity hash
            payload_sha256 = payload_hasher.digest()
            
            # Rewrite header with payload_sha256
            f.seek(96)
            f.write(payload_sha256)
        
        print(f"\n  Complete: {N:,} hashes written")
        print(f"  Payload SHA256: {payload_sha256.hex()[:32]}...")
        
        self.N = N
        return payload_sha256
    
    def build_prefix_index(self):
        """
        Step 3: Build prefix index files
        """
        print(f"\n[Indexing] Building prefix index (k={self.prefix_bytes})...")
        print(f"  Buckets: {self.buckets:,}")
        
        # Read all hashes
        with open(self.file_hashes, 'rb') as f:
            f.seek(self.HEADER_SIZE)
            hashes = f.read()
        
        # Build buckets
        buckets = [[] for _ in range(self.buckets)]
        
        for i in range(self.N):
            hash_bytes = hashes[i*32:(i+1)*32]
            
            # Calculate bucket
            bucket = int.from_bytes(hash_bytes[:self.prefix_bytes], 'big')
            buckets[bucket].append(i)
            
            if (i + 1) % 10000 == 0:
                print(f"  Bucketing: {i+1:,}/{self.N:,}", end='\r')
        
        print(f"\n  Buckets filled")
        
        # Write prefix_lists.bin
        print(f"  Writing {self.file_prefix_lists}...")
        lists_hasher = hashlib.sha256()
        
        with open(self.file_prefix_lists, 'wb') as f:
            for bucket_indices in buckets:
                for idx in bucket_indices:
                    # Use uint32 if N < 2^32, else uint64
                    if self.N < 2**32:
                        idx_bytes = struct.pack('<I', idx)
                    else:
                        idx_bytes = struct.pack('<Q', idx)
                    
                    f.write(idx_bytes)
                    lists_hasher.update(idx_bytes)
        
        lists_sha256 = lists_hasher.digest()
        
        # Write prefix_index.bin
        print(f"  Writing {self.file_prefix_idx}...")
        
        with open(self.file_prefix_idx, 'wb') as f:
            # Header (64 bytes)
            header = bytearray(64)
            header[0:8] = self.MAGIC_PREFIX
            struct.pack_into('<I', header, 8, self.VERSION)
            struct.pack_into('<I', header, 12, self.prefix_bytes)
            struct.pack_into('<Q', header, 16, self.N)
            struct.pack_into('<Q', header, 24, self.buckets)
            header[32:64] = lists_sha256
            f.write(header)
            
            # Bucket table
            start = 0
            for bucket_indices in buckets:
                count = len(bucket_indices)
                
                # Write (start, count)
                f.write(struct.pack('<Q', start))
                f.write(struct.pack('<Q', count))
                
                start += count
        
        print(f"  Index complete")
        print(f"  Lists SHA256: {lists_sha256.hex()[:32]}...")
    
    def verify_structural(self):
        """
        C1: Structural verification
        """
        print(f"\n[Verify] Structural integrity...")
        
        # File 1 size
        size1 = os.path.getsize(self.file_hashes)
        expected1 = self.HEADER_SIZE + self.N * 32
        
        if size1 != expected1:
            print(f"  ✗ File 1 size mismatch: {size1} != {expected1}")
            return False
        
        print(f"  ✓ File 1 size correct: {size1:,} bytes")
        
        # File 2 size
        size2 = os.path.getsize(self.file_prefix_idx)
        expected2 = 64 + self.buckets * 16
        
        if size2 != expected2:
            print(f"  ✗ File 2 size mismatch: {size2} != {expected2}")
            return False
        
        print(f"  ✓ File 2 size correct: {size2:,} bytes")
        
        return True
    
    def verify_cryptographic(self):
        """
        C2: Cryptographic integrity
        """
        print(f"\n[Verify] Cryptographic integrity...")
        
        # Verify payload_sha256
        with open(self.file_hashes, 'rb') as f:
            # Read stored hash
            f.seek(96)
            stored_hash = f.read(32)
            
            # Recalculate
            f.seek(self.HEADER_SIZE)
            payload = f.read()
            calculated_hash = hashlib.sha256(payload).digest()
        
        if stored_hash != calculated_hash:
            print(f"  ✗ Payload hash mismatch")
            return False
        
        print(f"  ✓ Payload integrity verified")
        return True
    
    def get_hash_by_index(self, index):
        """
        O(1) lookup: Get hash by index
        """
        if index < 0 or index >= self.N:
            raise IndexError(f"Index {index} out of range [0, {self.N})")
        
        with open(self.file_hashes, 'rb') as f:
            # Seek to hash position
            offset = self.HEADER_SIZE + index * 32
            f.seek(offset)
            return f.read(32)
    
    def find_by_prefix(self, prefix_bytes):
        """
        O(bucket_size) lookup: Find all hashes with given prefix
        Returns list of (index, hash) tuples
        """
        if len(prefix_bytes) != self.prefix_bytes:
            raise ValueError(f"Prefix must be {self.prefix_bytes} bytes")
        
        # Calculate bucket number
        bucket = int.from_bytes(prefix_bytes, 'big')
        
        # Read bucket metadata
        with open(self.file_prefix_idx, 'rb') as f:
            # Skip header (64 bytes) + seek to bucket entry
            offset = 64 + bucket * 16
            f.seek(offset)
            
            start = struct.unpack('<Q', f.read(8))[0]
            count = struct.unpack('<Q', f.read(8))[0]
        
        if count == 0:
            return []
        
        # Read indices from prefix_lists.bin
        results = []
        with open(self.file_prefix_lists, 'rb') as f:
            # Each index is 4 bytes (uint32) for N < 2^32
            idx_size = 4 if self.N < 2**32 else 8
            f.seek(start * idx_size)
            
            for _ in range(count):
                if idx_size == 4:
                    idx = struct.unpack('<I', f.read(4))[0]
                else:
                    idx = struct.unpack('<Q', f.read(8))[0]
                
                # Get the actual hash
                hash_val = self.get_hash_by_index(idx)
                results.append((idx, hash_val))
        
        return results
    
    def load_metadata(self):
        """
        Load database metadata from header
        """
        if not os.path.exists(self.file_hashes):
            raise FileNotFoundError(f"Database not found: {self.file_hashes}")
        
        with open(self.file_hashes, 'rb') as f:
            # Read header
            header = f.read(self.HEADER_SIZE)
            
            # Parse N
            self.N = struct.unpack('<Q', header[24:32])[0]
            
            print(f"[Loaded] Database with {self.N:,} entries")

