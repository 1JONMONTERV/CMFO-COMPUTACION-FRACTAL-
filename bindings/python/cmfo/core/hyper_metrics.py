"""
Hyper-Metrics 7D
================

Implements the 7-Dimensional Hyper-Resolution Manifold for CMFO states.
Reference: docs/theory/HYPER_7D_MANIFOLD.md
"""

import math
import numpy as np
from typing import Tuple, List
from .fractal_algebra_1_1 import FractalUniverse1024, NibbleAlgebra, Renormalization, Metrics

class HyperMetrics:
    
    @staticmethod
    def _entropy(probs: np.ndarray) -> float:
        """Standard Shannon Entropy in bits"""
        # Filter zeros
        p = probs[probs > 0]
        return -np.sum(p * np.log2(p))

    @staticmethod
    def _shannon_density(nibbles: np.ndarray) -> float:
        """D1: Information Density (Normalized Entropy 0..1)"""
        counts = np.bincount(nibbles, minlength=16)
        probs = counts / len(nibbles)
        h = HyperMetrics._entropy(probs)
        return h / 4.0 # Max entropy of nibble is 4 bits

    @staticmethod
    def _fractal_dimension(u: FractalUniverse1024) -> float:
        """D2: Fractal Scaling Dimension (Slope of Entropy decay)"""
        # Calculate entropy at 3 levels: 0, 1, 2
        # Level 0 (256 units)
        h0 = HyperMetrics._shannon_density(u.nibbles)
        
        # Level 1 (128 units)
        l1 = Renormalization.renorm_block_summary(u.nibbles)
        h1 = HyperMetrics._shannon_density(l1)
        
        # Level 2 (64 units)
        l2 = Renormalization.renorm_block_summary(l1)
        h2 = HyperMetrics._shannon_density(l2)
        
        # Slope of H vs Level?
        # If H decays, structure is simple. If H stays high, it's noise (D=1).
        # Linear fit to [h0, h1, h2] vs [0, 1, 2]
        # slope m. Generally H decreases.
        # Let's map to D: if slope is 0 (noise), D=1. If slope is steep (order), D=0.
        # Actually complexity: 
        # White noise: H stays ~1.0 at all scales.
        # Solid constant: H is 0.
        # Fractal: H decays with specific exponent.
        # We assign D2 = average H across 3 levels (Proxy for dimension/complexity).
        return (h0 + h1 + h2) / 3.0

    @staticmethod
    def _chirality(u: FractalUniverse1024) -> float:
        """D3: Mirror Asymmetry"""
        m = u.mirror()
        # Hamming distance between u and m
        # Nibble diffs
        # If u[i] == m[i], symmetry.
        # For M(n)=15-n, u[i]==m[i] is impossible for integers.
        # So we look at Distribution Symmetry?
        # NO, spec says d_H(x, M(x)).
        # Hamming distance normalized.
        # int diff normalized 0..15
        diff = np.sum(np.abs(u.nibbles.astype(int) - m.nibbles.astype(int)))
        # Max diff per nibble is 15. Total max is 15*256.
        return diff / (15 * 256)

    @staticmethod
    def _coherence(nibbles: np.ndarray) -> float:
        """D4: Spectral Coherence (1 - Entropy of Spectrum)"""
        # FFT of the sequence
        # Convert to float centered
        sig = nibbles.astype(float) - 7.5
        sp = np.fft.fft(sig)
        mag = np.abs(sp)
        # Normalize to probability
        tot = np.sum(mag)
        if tot < 1e-9: return 0.0 # DC only or silence
        probs = mag / tot
        h_spec = HyperMetrics._entropy(probs)
        # Max entropy of spectrum length 256 is log2(256)=8
        h_norm = h_spec / 8.0
        return 1.0 - h_norm # High coherence = Low entropy

    @staticmethod
    def _topological_charge(nibbles: np.ndarray) -> float:
        """D5: Defect Density (Transition density)"""
        # Simple diffs > 0
        diffs = np.diff(nibbles)
        transitions = np.count_nonzero(diffs)
        return transitions / (len(nibbles) - 1)

    @staticmethod
    def _octagonal_phase(nibbles: np.ndarray) -> float:
        """D6: Class Phase Orientation"""
        # Project each nibble to C8 (0..7) -> Angle
        # Sum vectors
        # c = kappa(C(n)). Here just n % 8 for simplicity or full Canon?
        # Let's use full Canon
        canons = [NibbleAlgebra.canon_4(n)[0] for n in nibbles]
        classes = [NibbleAlgebra.class_projection_8(c) for c in canons]
        
        # Convert to complex: exp(i * pi/4 * c)
        vectors = np.exp(1j * (np.pi/4) * np.array(classes))
        sum_vec = np.sum(vectors)
        
        # Angle normalized 0..1 (0..2pi)
        angle = np.angle(sum_vec)
        if angle < 0: angle += 2*np.pi
        return angle / (2*np.pi)

    @staticmethod
    def _singularity_potential(u: FractalUniverse1024) -> float:
        """D7: Distance to Zero (Null potential)"""
        # d_MS to Zero State
        # Just L1 norm here for speed as proxy
        # Sum of values
        s = np.sum(u.nibbles)
        max_s = 15 * 256
        # Normalized 'Energy'
        return s / max_s

    @staticmethod
    def compute_7d(u: FractalUniverse1024) -> np.ndarray:
        """Returns 7-element vector [v1..v7]"""
        v = np.zeros(7)
        v[0] = HyperMetrics._shannon_density(u.nibbles)
        v[1] = HyperMetrics._fractal_dimension(u)
        v[2] = HyperMetrics._chirality(u)
        v[3] = HyperMetrics._coherence(u.nibbles)
        v[4] = HyperMetrics._topological_charge(u.nibbles)
        v[5] = HyperMetrics._octagonal_phase(u.nibbles)
        v[6] = HyperMetrics._singularity_potential(u)
        return v

    @staticmethod
    def hyper_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Weighted Euclidean 7D Distance"""
        # Weights (can be tuned via PCA)
        w = np.ones(7)
        return np.sqrt(np.sum(w * (v1 - v2)**2))
