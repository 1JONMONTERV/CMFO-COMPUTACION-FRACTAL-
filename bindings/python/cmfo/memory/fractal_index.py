"""
Fractal Memory System 1.1
=========================

Implements Structural Memory using CMFO-FRACTAL-ALGEBRA 1.1.
- Indexing by Phi_90 Invariants.
- Geometric Search (d_MS).
- Anomaly Detection.

"""

import numpy as np
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from ..core.fractal_algebra_1_1 import FractalUniverse1024, Metrics, Renormalization

class FractalIndex:
    """
    In-memory structural index for 1024-bit states.
    """
    
    def __init__(self, resolution_bits=8):
        # Master storage: id -> FractalUniverse1024
        self.store: Dict[str, FractalUniverse1024] = {}
        
        # Structural Index: Phi90_Key -> List[id]
        self.index: Dict[str, List[str]] = {}
        
        # Parameters
        self.resolution = resolution_bits

    def _compute_key(self, x: FractalUniverse1024) -> str:
        """
        Generate structural key from Phi_90 map.
        Key = Hash(Quantize(Phi_90(x)))
        """
        phi_vec = Metrics.phi_90(x)
        
        # Quantize for indexing bucket
        # Simple quantization: Round to N decimals or binning
        # Since phi_vec contains std devs (0..15?), we can round to 0.1
        # For a "Structural Class", we want resonance.
        # Let's use a coarse quantization for the key (finding candidates)
        quantized = np.round(phi_vec * 10).astype(int)
        
        # Create hash of the structural vector
        key_str = quantized.tobytes()
        return hashlib.sha256(key_str).hexdigest()[:16] # 64-bit key prefix

    def add(self, data: bytes, item_id: str) -> str:
        """
        Add item to memory. 
        Returns structural key.
        """
        u = FractalUniverse1024(data)
        self.store[item_id] = u
        
        key = self._compute_key(u)
        
        if key not in self.index:
            self.index[key] = []
        self.index[key].append(item_id)
        
        return key

    def find_nearest(self, data: bytes, k=1) -> List[Tuple[str, float]]:
        """
        Find k nearest structural neighbors using d_MS.
        Optimized: Search in same bucket first, then expand?
        For this ref implementation: Scan all (O(N)).
        (Production would use BallTree or LSH on Phi vectors).
        """
        query_u = FractalUniverse1024(data)
        
        results = []
        for pid, u in self.store.items():
            dist = Metrics.distance_ms(query_u, u)
            results.append((pid, dist))
            
        # Sort by distance
        results.sort(key=lambda x: x[1])
        return results[:k]

    def find_resonance(self, data: bytes, threshold=1.0) -> List[str]:
        """
        Find items with distance < threshold.
        Uses Index for acceleration (Same Bucket Search).
        """
        query_u = FractalUniverse1024(data)
        key = self._compute_key(query_u)
        
        candidates = self.index.get(key, [])
        matches = []
        
        # Refine candidates
        for cid in candidates:
            u = self.store[cid]
            d = Metrics.distance_ms(query_u, u)
            if d < threshold:
                matches.append(cid)
                
        return matches

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_items": len(self.store),
            "structural_classes": len(self.index),
            "compression_ratio": len(self.store) / max(1, len(self.index))
        }
