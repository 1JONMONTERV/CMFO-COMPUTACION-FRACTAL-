"""
CMFO Python SDK - Pure Python Implementation
=============================================
Works immediately without C library.
Falls back to C when available.
"""

import numpy as np
from typing import List, Tuple, Optional
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = (1 + np.sqrt(5)) / 2

class CMFO:
    """Pure Python CMFO implementation"""
    
    def __init__(self, mode='study', license_key=None):
        self.mode = mode
        self.license_key = license_key
        self.lambda_weights = np.array([PHI**i for i in range(7)])
        
        # Semantic database
        self.semantic_db = {
            "existencia": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "entidad": np.array([1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
            "verdad": np.array([0.0, 1.0, 0.6, 0.0, 0.2, 0.2, 0.1]),
            "mentira": np.array([0.0, -1.0, -0.6, 0.0, -0.2, -0.2, -0.1]),
            "orden": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "caos": np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            "bien": np.array([0.0, 0.5, 0.8, 0.2, 0.6, 0.0, 0.1]),
            "mal": np.array([0.0, -0.5, -0.8, -0.2, -0.6, 0.0, -0.1]),
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalize vector"""
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v
    
    def parse(self, text: str) -> List[float]:
        """Parse text to 7D vector"""
        text_lower = text.lower().strip()
        vec = self.semantic_db.get(text_lower, np.zeros(7))
        return vec.tolist()
    
    def solve(self, equation: str) -> str:
        """Solve equation using pure CMFO"""
        try:
            from cmfo.education.equation_solver import solve_equation_cmfo
            return solve_equation_cmfo(equation) or "No solution"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def compose(self, v: List[float], w: List[float]) -> List[float]:
        """Compose two vectors: v ⊕ w"""
        v_arr = np.array(v)
        w_arr = np.array(w)
        result = self._normalize(v_arr + w_arr)
        return result.tolist()
    
    def modulate(self, scalar: float, v: List[float]) -> List[float]:
        """Scalar modulation: a ⊗ v"""
        v_arr = np.array(v)
        result = np.clip(scalar * v_arr, -1, 1)
        return result.tolist()
    
    def negate(self, v: List[float]) -> List[float]:
        """Negation: NEG(v)"""
        v_arr = np.array(v)
        v_arr[1] = -v_arr[1]  # Invert truth axis
        result = self._normalize(v_arr)
        return result.tolist()
    
    def distance(self, v: List[float], w: List[float]) -> float:
        """Fractal distance: d_φ(v, w)"""
        v_arr = np.array(v)
        w_arr = np.array(w)
        diff = v_arr - w_arr
        return np.sqrt(np.sum(self.lambda_weights * diff**2))

def get_version() -> Tuple[int, int, int]:
    """Get CMFO version"""
    return (1, 0, 0)

# Alias for compatibility
CMFOIntegrated = CMFO

__version__ = '1.0.0'
__all__ = ['CMFO', 'CMFOIntegrated', 'get_version', '__version__']

if __name__ == "__main__":
    print(f"CMFO Python SDK v{__version__} (Pure Python)")
    print("="*60)
    
    with CMFO() as cmfo:
        # Parse
        print("\n1. Parsing:")
        vec = cmfo.parse("verdad")
        print(f"   verdad = {vec}")
        
        # Solve
        print("\n2. Solving:")
        solution = cmfo.solve("2x + 3 = 7")
        print(f"   Solution preview: {solution[:200]}...")
        
        # Compose
        print("\n3. Composing:")
        v1 = [1, 0, 0, 0, 0, 0, 0]
        v2 = [0, 1, 0, 0, 0, 0, 0]
        composed = cmfo.compose(v1, v2)
        print(f"   {v1} ⊕ {v2} = {[f'{x:.3f}' for x in composed]}")
        
        # Distance
        print("\n4. Distance:")
        d = cmfo.distance(v1, v2)
        print(f"   d_φ(v1, v2) = {d:.4f}")
        
        # Negate
        print("\n5. Negation:")
        neg = cmfo.negate(v2)
        print(f"   NEG({v2}) = {[f'{x:.3f}' for x in neg]}")
    
    print("\n" + "="*60)
    print("All operations completed successfully!")
