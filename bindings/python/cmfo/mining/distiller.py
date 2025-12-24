"""
CMFO Mining - Fractal Block Distiller
=====================================

Implements the "Distillation" strategy for Proof-of-Space.
Instead of storing the entire universe of possible blocks (impossible),
we continuously scan sections of the 1024-bit block space and 
"distill" only the valid solutions into a dense database.

Key Features:
- 1024-bit Block Generation (Random Monte Carlo or Sequential).
- Fractal Hash Verification.
- High-Pass Filtering (Difficulty Target).
- Storage of "Golden Blocks" only.
"""

import numpy as np
import hashlib
import time
import json
import os

# We need the fractal hashing logic. 
# For this prototype, we simulate the "Fractal Hash" using SHA256d 
# but conceptualized as the geometric limit.
from ..constants import PHI

class BlockDistiller:
    def __init__(self, database_path='golden_db.json'):
        self.database_path = database_path
        self.solutions = []
        # Load existing if any
        if os.path.exists(database_path):
            try:
                with open(database_path, 'r') as f:
                    self.solutions = json.load(f)
            except:
                pass
        
    def fractal_hash(self, block_bytes):
        """
        Simulates the CMFO Fractal Hash (SHA256d + Phi-Mixing).
        Returns a hex string.
        """
        # 1. SHA256
        h1 = hashlib.sha256(block_bytes).digest()
        # 2. SHA256d (Standard Bitcoin-like) - this is our "Geometric Collapse"
        h2 = hashlib.sha256(h1).hexdigest()
        return h2

    def check_difficulty(self, hash_hex, target_zeros):
        """
        Checks if the hash meets the geometric difficulty (leading zeros).
        """
        return hash_hex.startswith('0' * target_zeros)

    def scan_batch(self, batch_size=10000, difficulty=4):
        """
        Generates and filters a batch of 1024-bit blocks.
        
        Args:
            batch_size: Number of blocks to process.
            difficulty: Number of leading zeros required (hex).
            
        Returns:
            int: Number of valid solutions found in this batch.
        """
        # Generate 1024-bit random blocks (128 bytes)
        # Using numpy for speed
        # random bytes: randint(0, 256)
        raw_data = np.random.randint(0, 256, (batch_size, 128), dtype=np.uint8)
        
        found_in_batch = 0
        
        for i in range(batch_size):
            block = raw_data[i].tobytes()
            h = self.fractal_hash(block)
            
            if self.check_difficulty(h, difficulty):
                # We found a Golden Block!
                # Store metadata
                solution = {
                    "block_hex": block.hex(), # Store full 1024-bit block
                    "hash": h,
                    "timestamp": time.time(),
                    "difficulty": difficulty
                }
                self.solutions.append(solution)
                found_in_batch += 1
                
        return found_in_batch

    def distiller_loop(self, total_blocks=1000000, difficulty=4):
        """
        Runs the distillation process for a requested number of blocks.
        """
        print(f"[Distiller] Starting scan of {total_blocks} blocks (Target: {difficulty} zeros)...")
        start_time = time.time()
        
        batch_size = min(10000, total_blocks)
        processed = 0
        
        while processed < total_blocks:
            found = self.scan_batch(batch_size, difficulty)
            processed += batch_size
            
            # Progress reporting
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  Scanned: {processed} | Found: {len(self.solutions)} | Speed: {rate:.0f} blocks/s", end='\r')
            
        print(f"\n[Distiller] Complete. Found {len(self.solutions)} Golden Blocks out of {total_blocks} visited.")
        self.save_db()
        
    def save_db(self):
        """Persist the Golden Database to disk."""
        with open(self.database_path, 'w') as f:
            json.dump(self.solutions, f, indent=2)
        print(f"[Storage] Saved {len(self.solutions)} solutions to {self.database_path}")

