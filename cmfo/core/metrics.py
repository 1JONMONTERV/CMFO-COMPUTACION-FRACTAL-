"""
CMFO Core Metrics
=================
Formal implementation of spec/metrics.md

Provides:
- d_φ: Phi-weighted Euclidean distance
- d_angle: Angular/cosine distance
- ~_ε: Regional equivalence relation
- Attractor detection
"""

import math
from typing import List, Tuple, Optional

PHI = 1.6180339887498948482


class CMFOMetrics:
    """Implements formal distance metrics from spec/metrics.md"""
    
    @staticmethod
    def d_phi(x: List[float], y: List[float]) -> float:
        """
        Phi-weighted Euclidean distance.
        
        d_φ(x,y) = √(Σᵢ φⁱ · (xᵢ - yᵢ)²)
        
        Args:
            x, y: 7D vectors
            
        Returns:
            Distance ≥ 0
        """
        if len(x) != 7 or len(y) != 7:
            raise ValueError("Vectors must be 7-dimensional")
        
        dist_sq = 0.0
        for i in range(7):
            weight = PHI ** i
            diff = x[i] - y[i]
            dist_sq += weight * diff * diff
        
        return math.sqrt(dist_sq)
    
    @staticmethod
    def d_angle(x: List[float], y: List[float]) -> float:
        """
        Angular distance (cosine-based).
        
        d_angle(x,y) = 1 - cos(θ) = 1 - (⟨x,y⟩_φ / (||x||_φ · ||y||_φ))
        
        Args:
            x, y: 7D vectors
            
        Returns:
            Distance in [0, 2]
        """
        if len(x) != 7 or len(y) != 7:
            raise ValueError("Vectors must be 7-dimensional")
        
        # Phi-weighted inner product
        inner_product = sum(PHI ** i * x[i] * y[i] for i in range(7))
        
        # Phi-weighted norms
        norm_x = math.sqrt(sum(PHI ** i * x[i] ** 2 for i in range(7)))
        norm_y = math.sqrt(sum(PHI ** i * y[i] ** 2 for i in range(7)))
        
        if norm_x < 1e-12 or norm_y < 1e-12:
            return 2.0  # Maximum distance for zero vectors
        
        cos_theta = inner_product / (norm_x * norm_y)
        # Clamp to [-1, 1] for numerical stability
        cos_theta = max(-1.0, min(1.0, cos_theta))
        
        return 1.0 - cos_theta
    
    @staticmethod
    def phi_norm(x: List[float]) -> float:
        """
        Phi-weighted norm.
        
        ||x||_φ = √(Σᵢ φⁱ · xᵢ²)
        """
        if len(x) != 7:
            raise ValueError("Vector must be 7-dimensional")
        
        return math.sqrt(sum(PHI ** i * x[i] ** 2 for i in range(7)))
    
    @staticmethod
    def regional_equiv(x: List[float], y: List[float], epsilon: float = 0.15) -> bool:
        """
        Regional equivalence relation.
        
        x ~_ε y  ⟺  d_φ(x,y) < ε
        
        Args:
            x, y: 7D vectors
            epsilon: Threshold (default 0.15 from spec)
            
        Returns:
            True if equivalent within epsilon
        """
        return CMFOMetrics.d_phi(x, y) < epsilon
    
    @staticmethod
    def normalize_phi(x: List[float]) -> List[float]:
        """
        Normalize to unit phi-norm.
        
        Γ_φ(x) = x / ||x||_φ
        """
        norm = CMFOMetrics.phi_norm(x)
        if norm < 1e-12:
            # Return zero vector if input is essentially zero
            return [0.0] * 7
        return [xi / norm for xi in x]


class AttractorDetector:
    """Implements attractor detection from spec/metrics.md"""
    
    def __init__(self, epsilon: float = 0.15):
        """
        Initialize attractor detector.
        
        Args:
            epsilon: Basin threshold (default 0.15)
        """
        self.epsilon = epsilon
        self.attractors: List[List[float]] = []
        self.basin_sizes: List[int] = []
    
    def find_attractor(self, vec: List[float]) -> Optional[Tuple[int, float]]:
        """
        Find nearest attractor within epsilon.
        
        Returns:
            (index, distance) if found, else None
        """
        best_idx = None
        best_dist = float('inf')
        
        for i, attractor in enumerate(self.attractors):
            dist = CMFOMetrics.d_phi(vec, attractor)
            if dist < self.epsilon and dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx is not None:
            return (best_idx, best_dist)
        return None
    
    def add_or_merge(self, vec: List[float]) -> Tuple[str, int]:
        """
        Add vector to attractor system.
        
        Returns:
            ('convergent', idx) or ('new_attractor', idx)
        """
        result = self.find_attractor(vec)
        
        if result is not None:
            idx, dist = result
            # Convergent: update attractor (moving average)
            old_count = self.basin_sizes[idx]
            new_count = old_count + 1
            
            old_attractor = self.attractors[idx]
            new_attractor = [
                (old_count * old_attractor[i] + vec[i]) / new_count
                for i in range(7)
            ]
            
            self.attractors[idx] = new_attractor
            self.basin_sizes[idx] = new_count
            
            return ('convergent', idx)
        else:
            # New attractor
            self.attractors.append(vec[:])  # Copy
            self.basin_sizes.append(1)
            return ('new_attractor', len(self.attractors) - 1)
    
    def get_stats(self) -> dict:
        """Get attractor statistics"""
        if not self.attractors:
            return {
                'num_attractors': 0,
                'total_points': 0,
                'convergence_rate': 0.0,
                'diversity': 1.0
            }
        
        total_points = sum(self.basin_sizes)
        convergent_points = total_points - len(self.attractors)
        
        return {
            'num_attractors': len(self.attractors),
            'total_points': total_points,
            'convergence_rate': convergent_points / total_points if total_points > 0 else 0.0,
            'diversity': len(self.attractors) / total_points if total_points > 0 else 1.0,
            'largest_basin': max(self.basin_sizes) if self.basin_sizes else 0,
            'median_basin': sorted(self.basin_sizes)[len(self.basin_sizes)//2] if self.basin_sizes else 0
        }
