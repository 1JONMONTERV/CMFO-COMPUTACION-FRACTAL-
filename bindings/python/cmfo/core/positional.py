"""
Positional Algebra (Coordinate Relativity)
==========================================

Implements the concept that Nibble Value is relative to its Position.
n_eff = (n + Delta(p)) mod 16

Reference: docs/theory/POSITIONAL_VALUE_THEORY.md
"""

import numpy as np
from typing import Callable, List
from .fractal_algebra_1_1 import FractalUniverse1024

class PositionalAlgebra:
    """
    Manages coordinate transformations of the Fractal Universe.
    """
    
    @staticmethod
    def delta_flat(size=256, shift=0) -> np.ndarray:
        """Delta(p) = Constant"""
        return np.full(size, shift, dtype=int)

    @staticmethod
    def delta_linear(size=256, slope=1) -> np.ndarray:
        """Delta(p) = p * slope (mod 16)"""
        # p=0..255
        idx = np.arange(size)
        return (idx * slope) % 16

    @staticmethod
    def delta_balanced(size=256) -> np.ndarray:
        """
        Anti-Symmetric Linear: Delta(p) + Delta(N-1-p) = 0.
        Satisfies Mirror Commutation.
        """
        # Map 0..255 to -128..127
        # Delta = (p - 127.5) -> scaled to nibble ring?
        # Let's try simple centering:
        idx = np.arange(size)
        # Shift so center is 0. 
        # 0 -> -8. 128 -> 0?
        # Let's map 256 steps to 16 steps? Or 1-to-1?
        # 1-to-1: Delta = p % 16.
        # Check symmetry:
        # p + (255-p) = 255 = -1 (mod 16). Sum is -1 != 0.
        # We need sum = 0 or 16.
        # Try Delta = p - 128 (mod 16). Same.
        # Try Delta[p] = p. Delta[N-1-p] = -p - 1. Sum = -1.
        # Fix: Add 0.5? No integers.
        # Construction: First half 0..127 define arbitrary. Second half -Delta.
        d = np.zeros(size, dtype=int)
        half = size // 2
        
        # Define first half as linear ramp 0..7, 0..7...
        # or just 0,1,2...
        ramp = np.arange(half) % 16
        d[:half] = ramp
        
        # Second half: -ramp reversed?
        # Delta(N-1-p) = -Delta(p)
        # let q = N-1-p. p=0 -> q=255.
        # d[255] = -d[0] = 0.
        # d[254] = -d[1] = -1 = 15.
        d[half:] = (-ramp[::-1]) % 16
        return d

    @staticmethod
    def delta_octagonal(size=256) -> np.ndarray:
        """Delta(p) based on 8-word structure of SHA state"""
        # 256 nibbles = 128 bytes = 32 words (4 bytes).
        # SHA state is 8 words. 
        # Block is 16 words (64 bytes) x 2 ? No block is 64 bytes (16 words).
        # 1024 bits = 128 bytes = 2 Blocks (64 bytes each).
        # Structure repeats every 64 nibbles (32 bytes)?
        # Let's try a period-64 ramp.
        idx = np.arange(size)
        return (idx // 8) % 8 # Word index mod 8?
        
    @staticmethod
    def apply(u: FractalUniverse1024, delta_arr: np.ndarray) -> FractalUniverse1024:
        """
        T(u) = u + Delta (mod 16)
        """
        if len(delta_arr) != len(u.nibbles):
            raise ValueError("Delta array size mismatch")
            
        new_nibs = (u.nibbles.astype(int) + delta_arr) % 16
        return FractalUniverse1024(new_nibs)

    @staticmethod
    def unapply(u: FractalUniverse1024, delta_arr: np.ndarray) -> FractalUniverse1024:
        """
        InvT(u) = u - Delta (mod 16)
        """
        if len(delta_arr) != len(u.nibbles):
            raise ValueError("Delta array size mismatch")
            
        new_nibs = (u.nibbles.astype(int) - delta_arr) % 16
        return FractalUniverse1024(new_nibs)
