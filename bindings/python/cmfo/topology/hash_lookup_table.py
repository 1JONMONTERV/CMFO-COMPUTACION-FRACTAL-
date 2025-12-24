"""
CMFO - Procedural Hash Lookup Table
====================================

Generates blocks from 2^512 space and computes their SHA256d hashes.
Creates a queryable table: Block → Hash

This enables "mining as lookup" by precomputing the hash space.
"""

import hashlib
import struct
import json
import os
from .procedural_512 import ProceduralSpace512

class HashLookupTable:
    """
    Procedural hash table for 2^512 space.
    Generates and indexes Block → SHA256d mappings.
    """
    
    def __init__(self, db_path='hash_lookup.json'):
        self.space = ProceduralSpace512()
        self.db_path = db_path
        self.table = {}
        
        # Load existing if present
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    loaded = json.load(f)
                    # Convert back from hex strings
                    self.table = {bytes.fromhex(k): v for k, v in loaded.items()}
                print(f"[DB] Loaded {len(self.table)} entries")
            except:
                pass
    
    def sha256d_fractal(self, block):
        """
        Computes SHA256d (double SHA-256) for a block.
        This is the standard Bitcoin hash function.
        
        Args:
            block: 64-byte (512-bit) block
            
        Returns:
            str: 64-character hex hash
        """
        # First SHA-256
        h1 = hashlib.sha256(block).digest()
        # Second SHA-256 (d = double)
        h2 = hashlib.sha256(h1).hexdigest()
        return h2
    
    def generate_entries(self, count=1000, start_x=0, start_y=0):
        """
        Generates hash table entries by sampling the 2^512 space.
        
        Args:
            count: Number of entries to generate
            start_x, start_y: Starting coordinates for sequential generation
        """
        print(f"\n[Generator] Creating {count} hash entries...")
        print(f"  Starting at: ({start_x}, {start_y})")
        
        generated = 0
        x, y = start_x, start_y
        
        for i in range(count):
            # Generate block from coordinates
            block = self.space.coords_to_block(x, y)
            
            # Compute hash
            hash_result = self.sha256d_fractal(block)
            
            # Store in table
            self.table[block] = hash_result
            
            # Progress
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{count}", end='\r')
            
            # Increment coordinates (sequential scan)
            y += 1
            if y >= 2**16:  # Wrap to next row
                y = 0
                x += 1
            
            generated += 1
        
        print(f"\n  Generated: {generated} entries")
        return generated
    
    def query_hash(self, block):
        """
        Looks up the hash for a given block.
        
        Args:
            block: 64-byte block
            
        Returns:
            str: Hash if found, None otherwise
        """
        return self.table.get(block)
    
    def query_by_coords(self, x, y):
        """
        Generates block from coords and returns its hash.
        """
        block = self.space.coords_to_block(x, y)
        
        # Check if already in table
        if block in self.table:
            return self.table[block], True  # (hash, cached)
        
        # Generate on-the-fly
        hash_result = self.sha256d_fractal(block)
        return hash_result, False  # (hash, not cached)
    
    def find_blocks_with_prefix(self, prefix, max_search=10000):
        """
        Searches for blocks whose hash starts with a specific prefix.
        This is the "mining" operation.
        
        Args:
            prefix: Hex string prefix (e.g., "0000" for 4 leading zeros)
            max_search: Maximum blocks to check
            
        Returns:
            list: Matching (block, hash) tuples
        """
        print(f"\n[Search] Looking for hashes starting with '{prefix}'...")
        print(f"  Max search: {max_search:,} blocks")
        
        matches = []
        x, y = 0, 0
        
        for i in range(max_search):
            block = self.space.coords_to_block(x, y)
            hash_result = self.sha256d_fractal(block)
            
            if hash_result.startswith(prefix):
                matches.append((block, hash_result))
                print(f"\n  ✓ Found: {hash_result[:32]}...")
            
            if (i + 1) % 1000 == 0:
                print(f"  Searched: {i+1:,} | Found: {len(matches)}", end='\r')
            
            # Next coordinate
            y += 1
            if y >= 2**16:
                y = 0
                x += 1
        
        print(f"\n  Total matches: {len(matches)}")
        return matches
    
    def save_table(self):
        """
        Persists the hash table to disk.
        """
        # Convert bytes keys to hex for JSON
        serializable = {k.hex(): v for k, v in self.table.items()}
        
        with open(self.db_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        file_size = os.path.getsize(self.db_path)
        print(f"\n[Save] Table saved: {self.db_path}")
        print(f"  Entries: {len(self.table):,}")
        print(f"  Size: {file_size/1024:.2f} KB")
