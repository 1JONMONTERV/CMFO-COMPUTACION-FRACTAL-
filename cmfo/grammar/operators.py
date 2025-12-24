"""
CMFO Grammar Operators
======================
9 fundamental grammatical operators implementing spec/algebra.md

All operators: Γ_φ(game(x,y;θ_op))
Unary operators: Fixed automorphisms
"""

import math
from typing import List, Tuple

# Constants
PHI = 1.6180339887498948

# Operator angles (frozen constants)
THETA_DET = 0.785  # π/4 - Determination
THETA_MOD = 1.047  # π/3 - Modification
THETA_REL = 0.524  # π/6 - Relation
THETA_APP_O = 1.571  # π/2 - Object application
THETA_APP_S = 0.785  # π/4 - Subject application
THETA_ADV = 0.628  # π/5 - Adverbial modification
THETA_NEG = 3.142  # π - Negation (rotation)
THETA_TENSE = 0.524  # π/6 - Temporal shift
THETA_COMP = 1.047  # π/3 - Comparison


def normalize_phi(v: List[float]) -> List[float]:
    """
    Normalize vector using φ-norm.
    Self-sufficient implementation without external dependencies.
    """
    # Calculate φ-weighted norm
    norm_sq = sum(v[i]**2 * (PHI ** i) for i in range(7))
    norm = math.sqrt(norm_sq)
    
    if norm < 1e-10:
        return [0.0] * 7
    
    # Normalize
    return [v[i] / norm for i in range(7)]


def game(x: List[float], y: List[float], theta: float) -> List[float]:
    """
    Base composition operator: game(x,y;θ) = Γ_φ(cos(θ)·x + sin(θ)·y)
    
    Args:
        x, y: 7D vectors
        theta: Mixing angle
        
    Returns:
        Normalized 7D vector
    """
    result = [
        math.cos(theta) * x[i] + math.sin(theta) * y[i]
        for i in range(7)
    ]
    return normalize_phi(result)


# ============================================================================
# BINARY OPERATORS
# ============================================================================

def DET(det: List[float], noun: List[float]) -> List[float]:
    """
    Determination: Det × N → N_det
    
    Examples: "el perro", "una casa"
    """
    return game(det, noun, THETA_DET)


def MOD_N(noun: List[float], adj: List[float]) -> List[float]:
    """
    Nominal Modification: N × A → N
    
    Examples: "perro negro", "casa grande"
    """
    return game(noun, adj, THETA_MOD)


def REL(noun: List[float], prep_phrase: List[float]) -> List[float]:
    """
    Relational: N × (P×N) → N_rel
    
    Examples: "casa de madera", "libro sobre física"
    Note: prep_phrase is already composed (P×N)
    """
    return game(noun, prep_phrase, THETA_REL)


def APP_O(verb_trans: List[float], obj: List[float]) -> List[float]:
    """
    Object Application: Vt × N → Vi
    
    Examples: "ve [a María]", "come [manzanas]"
    Produces verb with integrated object (Vi)
    """
    return game(verb_trans, obj, THETA_APP_O)


def APP_S(verb_intrans: List[float], subj: List[float]) -> List[float]:
    """
    Subject Application: Vi × N → S
    
    Examples: "[Juan] corre", "[El perro] come manzanas"
    Produces sentence (S)
    """
    return game(verb_intrans, subj, THETA_APP_S)


def ADV_V(verb: List[float], adverb: List[float]) -> List[float]:
    """
    Adverbial Modification: V × Adv → V
    
    Examples: "corre rápidamente", "habla claramente"
    """
    return game(verb, adverb, THETA_ADV)


def COMP(adj: List[float], noun1: List[float], noun2: List[float]) -> List[float]:
    """
    Comparison: A × N × N → S (binary chained)
    
    Examples: "Juan es más alto que María"
    Implementation: COMP(alto, Juan, María) = game(game(adj, n1), n2)
    """
    temp = game(adj, noun1, THETA_COMP)
    return game(temp, noun2, THETA_COMP)


# ============================================================================
# UNARY OPERATORS (Automorphisms)
# ============================================================================

def NEG_SCOPE(sentence: List[float]) -> List[float]:
    """
    Negation: S → S
    
    Examples: "Juan corre" → "Juan NO corre"
    Implements π-rotation in semantic space
    """
    # Rotation by π in first 2 dimensions (semantic polarity)
    result = sentence[:]
    result[0] = -sentence[0]  # Flip primary dimension
    result[1] = -sentence[1]  # Flip secondary dimension
    
    return normalize_phi(result)


def TENSE(sentence: List[float], tense_shift: float = THETA_TENSE) -> List[float]:
    """
    Temporal Operator: S → S
    
    Examples: "Juan corre" → "Juan corrió"
    Implements rotation in temporal dimensions (3,4)
    """
    result = sentence[:]
    
    # Rotate in temporal plane (dimensions 3,4)
    cos_t = math.cos(tense_shift)
    sin_t = math.sin(tense_shift)
    
    new_3 = cos_t * sentence[3] - sin_t * sentence[4]
    new_4 = sin_t * sentence[3] + cos_t * sentence[4]
    
    result[3] = new_3
    result[4] = new_4
    
    return normalize_phi(result)


# ============================================================================
# OPERATOR METADATA
# ============================================================================

OPERATORS = {
    'DET': {
        'func': DET,
        'signature': 'Det × N → N_det',
        'theta': THETA_DET,
        'arity': 2
    },
    'MOD_N': {
        'func': MOD_N,
        'signature': 'N × A → N',
        'theta': THETA_MOD,
        'arity': 2
    },
    'REL': {
        'func': REL,
        'signature': 'N × PP → N_rel',
        'theta': THETA_REL,
        'arity': 2
    },
    'APP_O': {
        'func': APP_O,
        'signature': 'Vt × N → Vi',
        'theta': THETA_APP_O,
        'arity': 2
    },
    'APP_S': {
        'func': APP_S,
        'signature': 'Vi × N → S',
        'theta': THETA_APP_S,
        'arity': 2
    },
    'ADV_V': {
        'func': ADV_V,
        'signature': 'V × Adv → V',
        'theta': THETA_ADV,
        'arity': 2
    },
    'NEG_SCOPE': {
        'func': NEG_SCOPE,
        'signature': 'S → S',
        'theta': THETA_NEG,
        'arity': 1
    },
    'TENSE': {
        'func': TENSE,
        'signature': 'S → S',
        'theta': THETA_TENSE,
        'arity': 1
    },
    'COMP': {
        'func': COMP,
        'signature': 'A × N × N → S',
        'theta': THETA_COMP,
        'arity': 3
    }
}


if __name__ == "__main__":
    # Quick sanity test
    print("CMFO Grammar Operators")
    print("=" * 60)
    
    for name, meta in OPERATORS.items():
        # ASCII-safe output
        sig = meta['signature'].replace('×', 'x').replace('→', '->')
        print(f"{name:12s} : {sig}")
    
    print("\nOperators loaded successfully.")
