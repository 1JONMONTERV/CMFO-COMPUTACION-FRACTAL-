"""
CMFO D11: Scientific Discovery Engine (Rule Derivation)
=======================================================
"CMFO no aprende por gradiente. Descubre por invariante."

Core Logic:
1. Observe pairs of states (State A -> State B).
2. Calculate transformation candidates (Delta = B - A).
3. Verify if Delta is constant (Invariant) across multiple samples.
4. If Invariant: Register as New Law.

Zero Training. One-Shot Discovery.
"""

import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Use D8 Algebra for vector calculations
try:
    from ..semantics.algebra import SemanticAlgebra, DIM
except ImportError:
    # Local fallback
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from semantics.algebra import SemanticAlgebra, DIM


@dataclass
class DiscoveredLaw:
    name: str # Auto-generated or labeled
    vector: List[float] # The Transform Vector
    confidence: float # 1.0 - variance
    samples: int # N observations


class RuleDiscoverer:
    """
    The Engine of Invariant Discovery.
    Detects algebraic patterns in data provided.
    """
    
    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance
        self.algebra = SemanticAlgebra()
    
    def analyze_pairs(self, pairs: List[Tuple[str, str]]) -> Optional[DiscoveredLaw]:
        """
        Analyze a list of (Input, Output) word pairs to find a common law.
        E.g. [("grande", "enorme"), ("bueno", "excelente")]
        """
        if not pairs:
            return None
            
        deltas = []
        
        # 1. Calculate Deltas (Displacements)
        for w1, w2 in pairs:
            v1 = self.algebra.value_of(w1)
            v2 = self.algebra.value_of(w2)
            
            # Law: v2 = v1 + delta  =>  delta = v2 - v1?
            # Or geometric projection? Ideally Vector Addition in this simplified algebra.
            # D8 Algebra uses normalized sums. This is non-linear inverse.
            # But for "Discovery" we look for additive displacement in the normalized space.
            
            delta = [b - a for a, b in zip(v1, v2)]
            deltas.append(delta)
            
        # 2. Check Invariance (Is delta constant?)
        avg_delta = [sum(x) / len(deltas) for x in zip(*deltas)]
        
        # Calculate Variance/Stability
        total_variance_sq = 0.0
        
        for d in deltas:
            dist_sq = sum((d[i] - avg_delta[i])**2 for i in range(DIM))
            total_variance_sq += dist_sq
            
        avg_variance = total_variance_sq / len(deltas)
        rmse = math.sqrt(avg_variance)
        
        print(f"[Discovery] Samples: {len(pairs)} | RMSE: {rmse:.4f}")
        
        # 3. Verdict
        if rmse < self.tolerance:
            # Law Discovered!
            # We normalize the delta to store it as a clean operator property?
            # Or keep it as a raw transformation vector.
            confidence = max(0.0, 1.0 - (rmse * 5)) # Heuristic scoring
            return DiscoveredLaw("discovered_op_01", avg_delta, confidence, len(pairs))
        else:
            return None

    def apply_law(self, law: DiscoveredLaw, word: str) -> List[float]:
        """Verify/Predict using the discovered law"""
        v1 = self.algebra.value_of(word)
        
        # Apply delta (Naive addition)
        v_pred_raw = [a + b for a, b in zip(v1, law.vector)]
        
        # Normalize (Always re-project to Unit Sphere in D8)
        return self.algebra.normalize(v_pred_raw)

if __name__ == "__main__":
    # Internal Test
    discoverer = RuleDiscoverer(tolerance=0.20)
    
    # Let's mock a "Very" operator if current Algebra lacks it
    # We rely on algebra.py definitions. 
    # If they don't capture recurrence, discovery will fail (correctly).
    # This requires 'enorme', 'excelente' to be defined in D8 Algebra first for valid demo!
    pass
