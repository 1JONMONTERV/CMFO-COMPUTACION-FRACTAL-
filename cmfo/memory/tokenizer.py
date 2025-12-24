"""
CMFO Fractal Tokenizer
======================
Maps discrete symbols (text/bytes) to points and trajectories on the T^7_phi manifold.

This is the bridge between discrete information (Language, Code) and 
continuous geometry (CMFO Physics).

Theory:
    Each byte/token corresponds to a discrete isometry or a 'move' in the space.
    A string is a trajectory X_0 -> X_1 -> ... -> X_N.
"""

import numpy as np
from typing import List, Union
from ..core.geometry import PHI, LAMBDA, wrap_angle

class FractalTokenizer:
    """
    Deterministic mapper from UTF-8 bytes to T^7 coordinates.
    """
    
    def __init__(self):
        # Initialize the "Codebook" - a set of 256 seed points uniform on T^7
        # In a real implementation, these would be optimized for separation energy.
        # Here we use a deterministic pseudo-random seed based on PHI.
        self._seed_points = self._generate_codebook()
        
    def _generate_codebook(self) -> np.ndarray:
        """Generates 256 deterministic points for byte mapping."""
        points = []
        for i in range(256):
            # Use phi-based chaos to generate deterministic points
            # formula: theta_k = (k * phi^dim) mod 2pi
            p = np.array([
                wrap_angle((i + 1) * (d + 1) * PHI**(d) * np.pi) 
                for d in range(7)
            ])
            points.append(p)
        return np.array(points)

    def encode_char(self, char: str) -> np.ndarray:
        """Encodes a single character/byte to a point in T^7."""
        byte_val = list(char.encode('utf-8'))[0] # Take first byte for simplicity in demo
        return self._seed_points[byte_val]

    def decode_point(self, point: np.ndarray) -> str:
        """
        Finds the nearest canonical point (token) to the given coordinate.
        This allows 'fuzzy' recovery of memory.
        """
        # Calculate distances to all 256 codebook points
        # Optimization: In production, use a KD-Tree or metric tree.
        dists = []
        for i in range(256):
            seed = self._seed_points[i]
            # Euclidean distance on the lift (simplification for speed)
            # Full geodesic_distance is O(1) but doing 256 times is linear.
            d = np.linalg.norm(point - seed)
            dists.append(d)
        
        nearest_idx = np.argmin(dists)
        return chr(nearest_idx) # Decode back to char (assuming ASCII/Single byte for demo)

    def text_to_trajectory(self, text: str) -> np.ndarray:
        """
        Converts text (sequence of N chars) to a trajectory tensor (N, 7).
        """
        traj = [self.encode_char(c) for c in text]
        return np.array(traj)

    def trajectory_to_text(self, trajectory: np.ndarray) -> str:
        """
        Recovers text from a trajectory.
        """
        chars = [self.decode_point(p) for p in trajectory]
        return "".join(chars)
