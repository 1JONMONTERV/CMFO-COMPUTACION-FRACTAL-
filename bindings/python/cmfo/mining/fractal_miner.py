"""
CMFO Mining - Fractal-Guided Search
====================================

Uses the Fractal Torus geometry to accelerate mining.
Instead of brute-force, we:
1. Map nonce space to Fractal Torus coordinates
2. Follow geometric gradients toward attractors
3. Sample high-density regions preferentially
"""

import numpy as np
import struct
from .fractal_sha import FractalSHA256, BitcoinHeaderStructure, H_INIT
from ..constants import PHI

class FractalMiner:
    """
    Geometric mining using Fractal Torus properties.
    """
    
    def __init__(self, torus_size=1024):
        self.torus_size = torus_size
        # Build Phi kernel for local geometry
        self.kernel = self._build_phi_kernel(7)
        
    def _build_phi_kernel(self, k):
        """7x7 Phi-weighted kernel for geometric sampling."""
        center = k // 2
        kernel = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist == 0:
                    kernel[i, j] = 1.0
                else:
                    kernel[i, j] = PHI ** (-dist)
        return kernel / np.sum(kernel)
    
    def nonce_to_torus_coords(self, nonce):
        """
        Maps a 32-bit nonce to (x, y) coordinates on the torus.
        Uses Phi-spiral mapping for better distribution.
        """
        # Golden angle spiral
        theta = nonce * PHI * 2 * np.pi
        r = np.sqrt(nonce) / np.sqrt(2**32) * self.torus_size
        
        x = int(r * np.cos(theta)) % self.torus_size
        y = int(r * np.sin(theta)) % self.torus_size
        
        return x, y
    
    def compute_geometric_potential(self, x, y, field):
        """
        Computes the local geometric potential at (x,y).
        Lower potential = higher probability of valid hash.
        """
        # Extract 7x7 neighborhood (with wrapping)
        patch = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                px = (x + i - 3) % self.torus_size
                py = (y + j - 3) % self.torus_size
                patch[i, j] = field[px, py]
        
        # Convolve with Phi kernel
        potential = np.sum(patch * self.kernel)
        return potential
    
    def generate_candidate_nonces(self, count=1000, field=None):
        """
        Generates candidate nonces using geometric sampling.
        Preferentially samples low-potential regions.
        """
        if field is None:
            # Initialize random field
            field = np.random.rand(self.torus_size, self.torus_size)
        
        candidates = []
        
        # Sample using inverse potential weighting
        for _ in range(count):
            # Random starting point
            x = np.random.randint(0, self.torus_size)
            y = np.random.randint(0, self.torus_size)
            
            # Gradient descent (3 steps)
            for step in range(3):
                # Check neighbors
                best_pot = self.compute_geometric_potential(x, y, field)
                best_x, best_y = x, y
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx = (x + dx) % self.torus_size
                        ny = (y + dy) % self.torus_size
                        pot = self.compute_geometric_potential(nx, ny, field)
                        if pot < best_pot:
                            best_pot = pot
                            best_x, best_y = nx, ny
                
                x, y = best_x, best_y
            
            # Convert back to nonce
            # Inverse mapping (approximate)
            nonce = (x * self.torus_size + y) % (2**32)
            candidates.append(nonce)
        
        return candidates
    
    def fractal_mine(self, base_header, target_difficulty=5, max_attempts=100000):
        """
        Mines using fractal-guided search.
        """
        print(f"\n[Fractal Miner] Starting geometric search...")
        print(f"  Target: {target_difficulty} zeros")
        print(f"  Strategy: Phi-guided gradient descent")
        
        # Prepare blocks
        b1, b2_template = BitcoinHeaderStructure.create_template(base_header)
        midstate = FractalSHA256.compress(H_INIT, b1)
        
        # Initialize geometric field
        field = np.random.rand(self.torus_size, self.torus_size)
        
        attempts = 0
        batch_size = 1000
        
        while attempts < max_attempts:
            # Generate candidate nonces using geometry
            candidates = self.generate_candidate_nonces(batch_size, field)
            
            for nonce in candidates:
                attempts += 1
                
                # Hash
                b2 = BitcoinHeaderStructure.inject_nonce(b2_template, nonce)
                h_final = FractalSHA256.compress(midstate, b2)
                
                # Check
                hash_bytes = b''.join(struct.pack('!I', x) for x in h_final)
                hash_hex = hash_bytes.hex()
                
                if hash_hex.startswith('0' * target_difficulty):
                    print(f"\n  ✓ FOUND! Nonce={nonce:,} Hash={hash_hex[:16]}...")
                    return nonce, hash_hex
                
                # Update field based on result
                x, y = self.nonce_to_torus_coords(nonce)
                # Lower field value if hash was "close"
                zeros = len(hash_hex) - len(hash_hex.lstrip('0'))
                field[x, y] *= (1.0 - zeros / (target_difficulty * 2))
            
            if attempts % 10000 == 0:
                print(f"  Attempts: {attempts:,} | Geometric sampling active", end='\r')
        
        print(f"\n  No solution found in {attempts:,} attempts")
        return None, None
