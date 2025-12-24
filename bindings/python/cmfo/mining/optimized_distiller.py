"""
CMFO Mining - Optimized Block Distiller (Midstate)
==================================================

High-performance distiller using Midstate optimization.
Builds a comprehensive database of valid mining solutions.
"""

import struct
import time
import json
import os
from .fractal_sha import FractalSHA256, BitcoinHeaderStructure, H_INIT

class OptimizedDistiller:
    """
    Scans the nonce space using Midstate optimization.
    Only stores blocks that meet the difficulty target.
    """
    
    def __init__(self, database_path='golden_blocks_db.json'):
        self.database_path = database_path
        self.solutions = []
        
        # Load existing database if present
        if os.path.exists(database_path):
            try:
                with open(database_path, 'r') as f:
                    self.solutions = json.load(f)
                print(f"[DB] Loaded {len(self.solutions)} existing solutions")
            except:
                pass
    
    def create_base_header(self, version=2, prev_hash=None, merkle_root=None, 
                          timestamp=None, bits=None):
        """
        Creates a base 80-byte Bitcoin header template.
        """
        header = bytearray(80)
        
        # Version (4 bytes, little-endian)
        struct.pack_into('<I', header, 0, version)
        
        # Previous block hash (32 bytes)
        if prev_hash:
            header[4:36] = prev_hash
        else:
            header[4:36] = b'\xaa' * 32  # Dummy
        
        # Merkle root (32 bytes)
        if merkle_root:
            header[36:68] = merkle_root
        else:
            header[36:68] = b'\xff' * 32  # Dummy
        
        # Timestamp (4 bytes, little-endian)
        if timestamp:
            struct.pack_into('<I', header, 68, timestamp)
        else:
            struct.pack_into('<I', header, 68, int(time.time()))
        
        # Bits/Difficulty (4 bytes, little-endian)
        if bits:
            struct.pack_into('<I', header, 72, bits)
        else:
            struct.pack_into('<I', header, 72, 0x1d00ffff)  # Standard difficulty
        
        # Nonce (4 bytes) - will be varied
        struct.pack_into('<I', header, 76, 0)
        
        return bytes(header)
    
    def check_difficulty(self, hash_state, target_zeros):
        """
        Checks if the hash meets difficulty (leading zeros).
        hash_state: list of 8 uint32 values
        """
        # Convert to bytes (big-endian for display)
        hash_bytes = b''.join(struct.pack('!I', x) for x in hash_state)
        hash_hex = hash_bytes.hex()
        
        return hash_hex.startswith('0' * target_zeros), hash_hex
    
    def scan_nonce_space(self, base_header, start_nonce=0, count=1000000, 
                        difficulty=4, batch_report=50000):
        """
        Scans a range of nonces using Midstate optimization.
        
        Args:
            base_header: 80-byte header template
            start_nonce: Starting nonce value
            count: Number of nonces to try
            difficulty: Number of leading hex zeros required
            batch_report: Report progress every N iterations
        """
        print(f"\n[Scanner] Starting nonce scan...")
        print(f"  Range: {start_nonce:,} to {start_nonce + count:,}")
        print(f"  Target: {difficulty} leading zeros")
        
        # Prepare blocks
        b1, b2_template = BitcoinHeaderStructure.create_template(base_header)
        
        # Compute Midstate (H1) - ONCE
        print(f"[Midstate] Computing H1...")
        midstate = FractalSHA256.compress(H_INIT, b1)
        print(f"[Midstate] H1 = {' '.join(f'{x:08x}' for x in midstate[:4])}...")
        
        # Scan loop
        start_time = time.time()
        found_count = 0
        
        for i in range(count):
            nonce = start_nonce + i
            
            # Inject nonce into B2
            b2 = BitcoinHeaderStructure.inject_nonce(b2_template, nonce)
            
            # Compute final hash from midstate
            h_final = FractalSHA256.compress(midstate, b2)
            
            # Check difficulty
            valid, hash_hex = self.check_difficulty(h_final, difficulty)
            
            if valid:
                # Found a golden block!
                solution = {
                    "header_hex": base_header.hex(),
                    "nonce": nonce,
                    "hash": hash_hex,
                    "difficulty": difficulty,
                    "timestamp": time.time()
                }
                self.solutions.append(solution)
                found_count += 1
                print(f"\n  ✓ FOUND #{found_count}: Nonce={nonce:,} Hash={hash_hex[:16]}...")
            
            # Progress report
            if (i + 1) % batch_report == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i+1:,}/{count:,} | Rate: {rate:.0f} h/s | Found: {found_count}", end='\r')
        
        elapsed = time.time() - start_time
        final_rate = count / elapsed
        
        print(f"\n\n[Scanner] Complete!")
        print(f"  Total Scanned: {count:,}")
        print(f"  Solutions Found: {found_count}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Average Rate: {final_rate:.0f} hashes/second")
        
        return found_count
    
    def save_database(self):
        """Persist solutions to disk."""
        with open(self.database_path, 'w') as f:
            json.dump(self.solutions, f, indent=2)
        
        file_size = os.path.getsize(self.database_path)
        print(f"\n[DB] Saved {len(self.solutions)} solutions")
        print(f"[DB] File: {self.database_path}")
        print(f"[DB] Size: {file_size/1024:.2f} KB")
    
    def query_solution(self, header_prefix=None):
        """
        Query the database for a matching solution.
        Returns the first matching nonce.
        """
        if header_prefix:
            for sol in self.solutions:
                if sol['header_hex'].startswith(header_prefix):
                    return sol
        
        # Return any solution
        if self.solutions:
            return self.solutions[0]
        return None
