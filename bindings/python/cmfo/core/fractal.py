"""
CMFO Fractal Algebra Core
==========================

Complete implementation of the CMFO algebraic framework:
- Fractal field operations (⊕φ, ⊗φ, ℛφ)
- φ-logic operators
- Deterministic decision functions

Reference: docs/theory/CMFO_COMPLETE_ALGEBRA.md
"""

import math
from typing import Union, List

# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


# ============================================================================
# I. FRACTAL FIELD OPERATIONS
# ============================================================================

def fractal_root(x: Union[float, List[float]]) -> Union[float, List[float]]:
    """
    The fundamental CMFO operator: ℛφ(x) = x^(1/φ)
    
    Collapses hierarchical structures to their geometric core.
    
    Properties:
    - ℛφ(φ^k) = φ^(k/φ)
    - lim_{n→∞} ℛφ^(n)(x) = 1
    - Non-linear: ℛφ(x+y) ≠ ℛφ(x) + ℛφ(y)
    
    Args:
        x: Positive real number or list
        
    Returns:
        Fractally collapsed value
    """
    if isinstance(x, (list, tuple)):
        return [val ** PHI_INV for val in x]
    return x ** PHI_INV


def fractal_add(x: float, y: float) -> float:
    """
    Fractal addition: x ⊕φ y = x + y
    
    Standard addition—fractality emerges from other operations.
    """
    return x + y


def fractal_multiply(x: float, y: float) -> float:
    """
    Fractal multiplication: x ⊗φ y = x^(log_φ y)
    
    Encodes scale, not quantity.
    """
    if y <= 0:
        raise ValueError("Fractal multiplication requires positive arguments")
    return x ** (math.log(y) / math.log(PHI))


# ============================================================================
# II. φ-LOGIC OPERATORS
# ============================================================================

class PhiBit:
    """
    Fractal bit (φ-bit): b_φ ∈ {φ^(-1), 1, φ}
    
    - φ^(-1) ≈ 0.618 → Structural False
    - 1             → Neutral
    - φ ≈ 1.618     → Structural True
    """
    FALSE = PHI_INV
    NEUTRAL = 1.0
    TRUE = PHI
    
    @staticmethod
    def from_bool(b: bool) -> float:
        """Convert boolean to φ-bit"""
        return PhiBit.TRUE if b else PhiBit.FALSE
    
    @staticmethod
    def to_bool(phi_bit: float, threshold: float = 1.0) -> bool:
        """Convert φ-bit to boolean"""
        return phi_bit > threshold


def phi_and(a: float, b: float) -> float:
    """
    Fractal AND: a ∧φ b = ℛφ(a · b)
    
    Geometric conjunction with stability.
    """
    return fractal_root(a * b)


def phi_or(a: float, b: float) -> float:
    """
    Fractal OR: a ∨φ b = ℛφ(a + b)
    
    Geometric disjunction with stability.
    """
    return fractal_root(a + b)


def phi_not(a: float) -> float:
    """
    Fractal NOT: ¬φ a = φ / a
    
    Geometric negation.
    """
    return PHI / a


# ============================================================================
# III. DETERMINISTIC DECISION FUNCTIONS
# ============================================================================

def phi_normalize(x: List[float]) -> List[float]:
    """
    Fractal normalization: replaces softmax and L2 normalization.
    
    Each component is collapsed to its geometric core, then renormalized.
    """
    collapsed = [fractal_root(abs(val)) for val in x]
    total = sum(collapsed)
    return [val / total for val in collapsed]


def phi_activation(x: List[float]) -> List[float]:
    """
    Fractal activation function for neural networks.
    
    Replaces ReLU, sigmoid, tanh with geometric collapse.
    """
    return [fractal_root(max(val, PHI_INV)) for val in x]


def phi_decision(logits: List[float]) -> int:
    """
    Deterministic decision from logits (no softmax, no sampling).
    
    Returns the index of the geometrically dominant component.
    
    Args:
        logits: Raw network outputs
        
    Returns:
        Index of selected class (deterministic)
    """
    normalized = phi_normalize(logits)
    return max(range(len(normalized)), key=lambda i: normalized[i])


# ============================================================================
# IV. GEOMETRIC COLLAPSE (PHYSICS)
# ============================================================================

def geometric_state_collapse(psi: List[complex]) -> float:
    """
    Geometric quantum state collapse: ψ_real = ℛφ(Σ|ψ_i|²)
    
    The state collapses through geometry, not through an observer.
    
    Args:
        psi: Quantum state vector (complex amplitudes)
        
    Returns:
        Collapsed real value
    """
    probabilities = [abs(p) ** 2 for p in psi]
    return fractal_root(sum(probabilities))


def fractal_time_flow(velocity_norm: float) -> float:
    """
    Time as fractal root of flow: dτ = ℛφ(||Ẋ||_g)
    
    Time emerges from geometric flow.
    
    Args:
        velocity_norm: Norm of velocity vector
        
    Returns:
        Proper time differential
    """
    return fractal_root(velocity_norm)
