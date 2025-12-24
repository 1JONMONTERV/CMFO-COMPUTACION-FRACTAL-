"""
CMFO Fractal Intermediate Representation (IR)
=============================================
Defines the semantic graph structure for 7D Geometric Operators.
This is the "structural truth" that the compiler optimizes.
"""

from dataclasses import dataclass
from typing import List, Union, Optional

@dataclass
class FractalNode:
    """Base class for all IR nodes"""
    name: Optional[str] = None
    
    def __repr__(self):
        return self.__class__.__name__

@dataclass
class Symbol(FractalNode):
    """Represents an input tensor (Argument)"""
    shape: tuple = (7,)
    
    def __repr__(self):
        return f"Symbol({self.name})"

@dataclass
class Constant(FractalNode):
    """Represents a scalar constant (e.g., PHI)"""
    value: float = 0.0
    
    def __repr__(self):
        return f"Const({self.value:.4f})"

@dataclass
class AlgebraicOp(FractalNode):
    """
    Element-wise algebraic operations.
    Fused by the 'Sniper' kernel.
    """
    op_type: str = "nop"
    left: FractalNode = None
    right: FractalNode = None

    def __repr__(self):
        return f"({self.left} {self.op_type} {self.right})"

@dataclass
class GeometricOp(FractalNode):
    """
    7D Manifold Operations (The specific domain of CMFO)
    """
    op_type: str = "nop"
    input_node: FractalNode = None
    params: dict = None

    def __repr__(self):
        return f"{self.op_type}({self.input_node})"

# --- Builder Helpers ---

def symbol(name):
    return Symbol(name=name)

def constant(val):
    return Constant(value=val)

def fractal_mul(a, b):
    return AlgebraicOp(op_type='mul', left=a, right=b)

def fractal_add(a, b):
    return AlgebraicOp(op_type='add', left=a, right=b)

def fractal_div(a, b):
    return AlgebraicOp(op_type='div', left=a, right=b)

def fractal_root(a):
    return AlgebraicOp(op_type='pow', left=a, right=constant(0.618034)) # x^(1/PHI)

def gamma_step(state):
    return GeometricOp(op_type='gamma', input_node=state)

def fractal_sub(a, b):
    return AlgebraicOp(op_type='sub', left=a, right=b)

def algebraic_op(op, left, right):
    return AlgebraicOp(op_type=op, left=left, right=right)

def geometric_op(op, node, params=None):
    return GeometricOp(op_type=op, input_node=node, params=params)

def fractal_min(a, b):
    # For fuzzy/fractal logic: min(a,b) is AND
    return AlgebraicOp(op_type='min', left=a, right=b)

def fractal_step(a):
    # Heaviside step / Sign function
    return GeometricOp(op_type='step', input_node=a)

def fractal_sqrt(a):
    return GeometricOp(op_type='sqrt', input_node=a)

def fractal_sin(a):
    return GeometricOp(op_type='sin', input_node=a)

def fractal_pow(a, b):
    """Operación de potencia genérica: a^b"""
    return AlgebraicOp(op_type='pow', left=a, right=b)

