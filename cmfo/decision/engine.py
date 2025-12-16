"""
CMFO Decision Engine
====================
Geometric decision function over semantic basins.

NOT probabilistic generation - deterministic state selection.

D(S_input, M_memory, C_context) -> S_response
"""

import math
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


PHI = 1.6180339887


@dataclass
class SemanticState:
    """Semantic state in T^7_φ"""
    vector: List[float]
    label: str  # Human-readable label
    type: str   # 'affirmation', 'question', 'correction', etc.


@dataclass
class Memory:
    """Fractal memory (persistent states)"""
    states: List[SemanticState]
    attractors: List[List[float]]  # Dominant basins
    
    def add(self, state: SemanticState):
        """Add state to memory"""
        self.states.append(state)
    
    def recent(self, n: int = 5) -> List[SemanticState]:
        """Get n most recent states"""
        return self.states[-n:]


def d_phi(x: List[float], y: List[float]) -> float:
    """Phi-weighted distance"""
    dist_sq = 0.0
    for i in range(7):
        weight = PHI ** i
        diff = x[i] - y[i]
        dist_sq += weight * diff * diff
    return math.sqrt(dist_sq)


def phi_norm(x: List[float]) -> float:
    """Phi-weighted norm"""
    return math.sqrt(sum(PHI ** i * x[i] ** 2 for i in range(7)))


class DecisionEngine:
    """
    Geometric decision engine.
    
    Selects semantic states via optimization, not probability.
    """
    
    def __init__(self, 
                 alpha: float = 1.0,  # Weight for input similarity
                 beta: float = 0.5,   # Weight for memory coherence
                 gamma: float = 0.1): # Weight for cost
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Response templates (semantic states, not text)
        self.response_templates = self._build_templates()
    
    def _build_templates(self) -> Dict[str, SemanticState]:
        """
        Build response templates.
        
        These are NOT text templates - they are semantic states
        representing different response types.
        """
        # Placeholder vectors (in real system, these would be learned/calibrated)
        return {
            'confirm': SemanticState(
                vector=[0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
                label='confirmation',
                type='affirmation'
            ),
            'correct': SemanticState(
                vector=[-0.8, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0],
                label='correction',
                type='correction'
            ),
            'question': SemanticState(
                vector=[0.0, 0.0, 0.8, 0.5, 0.2, 0.0, 0.0],
                label='question',
                type='question'
            ),
            'reference': SemanticState(
                vector=[0.5, 0.5, 0.0, 0.0, 0.5, 0.3, 0.0],
                label='memory_reference',
                type='reference'
            ),
            'conflict': SemanticState(
                vector=[-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                label='contradiction_signal',
                type='conflict'
            )
        }
    
    def cost(self, state: SemanticState) -> float:
        """
        Cost of producing this response.
        
        Lower cost = simpler/more direct response.
        """
        costs = {
            'affirmation': 0.1,  # Cheap
            'correction': 0.5,   # Moderate
            'question': 0.3,     # Moderate
            'reference': 0.4,    # Moderate
            'conflict': 0.8      # Expensive (serious)
        }
        return costs.get(state.type, 1.0)
    
    def decide_response(self,
                       S_input: List[float],
                       memory: Memory,
                       context: Optional[Dict] = None) -> Tuple[SemanticState, Dict]:
        """
        Decide response via geometric optimization.
        
        Args:
            S_input: Input semantic state
            memory: Fractal memory
            context: Optional context dict
            
        Returns:
            (chosen_state, metrics)
        """
        candidates = list(self.response_templates.values())
        
        best_state = None
        best_score = float('inf')
        metrics = {}
        
        # Get recent memory for coherence
        recent_states = memory.recent(n=3)
        avg_memory = self._average_states([s.vector for s in recent_states]) if recent_states else None
        
        for candidate in candidates:
            # Distance to input
            d_input = d_phi(candidate.vector, S_input)
            
            # Distance to memory (coherence)
            d_memory = d_phi(candidate.vector, avg_memory) if avg_memory else 0.0
            
            # Cost
            c = self.cost(candidate)
            
            # Total score (minimize)
            score = self.alpha * d_input + self.beta * d_memory + self.gamma * c
            
            if score < best_score:
                best_score = score
                best_state = candidate
                metrics = {
                    'd_input': d_input,
                    'd_memory': d_memory,
                    'cost': c,
                    'total_score': score
                }
        
        return best_state, metrics
    
    def _average_states(self, vectors: List[List[float]]) -> List[float]:
        """Average multiple vectors"""
        if not vectors:
            return [0.0] * 7
        
        avg = [0.0] * 7
        for vec in vectors:
            for i in range(7):
                avg[i] += vec[i]
        
        for i in range(7):
            avg[i] /= len(vectors)
        
        return avg
    
    def classify_input(self, S_input: List[float], memory: Memory) -> str:
        """
        Classify input type based on geometry.
        
        Returns: 'correct', 'false', 'ambiguous', 'repetition', 'contradiction'
        """
        # Check against memory attractors
        if memory.attractors:
            min_dist = min(d_phi(S_input, attr) for attr in memory.attractors)
            
            if min_dist < 0.1:
                return 'repetition'  # Very close to known state
            elif min_dist > 1.0:
                return 'contradiction'  # Far from all known states
        
        # Check norm (ambiguity heuristic)
        norm = phi_norm(S_input)
        if norm < 0.5:
            return 'ambiguous'
        
        # Check polarity (correct vs false heuristic)
        if S_input[0] > 0:
            return 'correct'
        else:
            return 'false'


if __name__ == "__main__":
    print("CMFO Decision Engine")
    print("=" * 60)
    
    # Test
    engine = DecisionEngine()
    memory = Memory(states=[], attractors=[])
    
    # Example input
    S_input = [0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]
    
    response, metrics = engine.decide_response(S_input, memory)
    
    print(f"\nInput: {S_input[:3]}...")
    print(f"Chosen: {response.label}")
    print(f"Type: {response.type}")
    print(f"Metrics: {metrics}")
    print("\nDecision engine loaded successfully.")
