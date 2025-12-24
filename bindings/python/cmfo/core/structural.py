"""
CMFO Structural Algebra
=======================
Pure Python implementation of 7-Dimensional Geometric Algebra.
Zero external dependencies (No NumPy).
"""

import math
import cmath
from ..compiler.ir import FractalNode, symbol, constant, fractal_add, fractal_sub, fractal_mul, algebraic_op, geometric_op
from ..compiler.jit import FractalJIT

class FractalVector7:
    """
    A 7-dimensional vector in the fractal manifold.
    Supports TWO modes:
    1. Eager (Standard): Stored as list of complex numbers. Calculated on CPU.
    2. Lazy (JIT): Stored as an IR Graph Node. Calculated on GPU ('compute()').
    """
    __slots__ = ['v', '_node', '_check_inputs']

    def __init__(self, values=None, node=None):
        self._node = node
        self._check_inputs = [] # Stores (data_vector) to satisfy graph leaves
        
        if node is not None:
             self.v = None # Lazy mode
        else:
            if values is None:
                self.v = [0j] * 7
            else:
                if len(values) != 7:
                    raise ValueError("FractalVector7 must have exactly 7 components")
                self.v = list(values)

    @property
    def is_lazy(self):
        return self._node is not None

    def __repr__(self):
        if self.is_lazy:
            return f"FractalVector7(LazyNode={self._node})"
        formatted = ", ".join(f"{x:.2f}" for x in self.v)
        return f"FractalVector7([{formatted}])"

    def compute(self, input_h=None):
        """
        Triggers Native JIT compilation and execution.
        Currently maps 'self' history to input_h if provided.
        Only supports simple unary/binary evolution for now.
        """
        if not self.is_lazy:
            return self
        
        # 1. Gather Inputs
        # For this Beta, we assume the graph is built on top of a 'root' vector
        # passed as symbolic 'v' in the IR.
        # We need the ACTUAL data for 'v'.
        # Since we don't have a full dependency injection system yet,
        # we require the user to provide the 'context' data or we use the cached inputs in _check_inputs (future).
        
        # HACK for Demo: We assume the user creates a "Symbolic Vector" first
        # v_sym = FractalVector7(node=symbol('v')) -> holds no data
        # To run, we need data.
        
        raise NotImplementedError("Auto-JIT Compute requires Data Context binding (Coming in v3.2)")

    @classmethod
    def symbolic(cls, name='v'):
        """Create a purely symbolic vector for graph construction"""
        return cls(node=symbol(name))

    def __add__(self, other):
        # JIT Path
        if self.is_lazy or (isinstance(other, FractalVector7) and other.is_lazy) or self.is_lazy:
            left = self._node if self.is_lazy else constant(0.0) # Placeholder for non-lazy vector injection (Not supported yet)
            
            if isinstance(other, FractalVector7):
                 right = other._node if other.is_lazy else constant(0.0)
            elif isinstance(other, (int, float)):
                 right = constant(other)
            else:
                 right = constant(0.0)
                 
            return FractalVector7(node=fractal_add(left, right))
            
        return FractalVector7([a + b for a, b in zip(self.v, other.v)])

    def __sub__(self, other):
        # JIT Path
        if self.is_lazy or (isinstance(other, FractalVector7) and other.is_lazy) or self.is_lazy:
            left = self._node if self.is_lazy else constant(0.0)
            
            if isinstance(other, FractalVector7):
                 right = other._node if other.is_lazy else constant(0.0)
            elif isinstance(other, (int, float)):
                 right = constant(other)
            else:
                 right = constant(0.0)

            return FractalVector7(node=fractal_sub(left, right))

        return FractalVector7([a - b for a, b in zip(self.v, other.v)])

    def norm(self):
        """Euclidean norm (L2)"""
        if self.is_lazy:
             # Norm is a reduction, JIT output is scalar? Currently JIT returns 7D.
             # We would need a 'reduce' kernel.
             raise NotImplementedError("JIT Norm not supported")
             
        sum_sq = sum(abs(x)**2 for x in self.v)
        return math.sqrt(sum_sq)

    def normalize(self):
        """Return a normalized copy of the vector"""
        if self.is_lazy:
             # Algebraic normalize in IR?
             # return self / self.norm()
             pass 
             
        m = self.norm()
        if m < 1e-15:
            return FractalVector7(self.v) # Return zero vec if too small
        inv_m = 1.0 / m
        return FractalVector7([x * inv_m for x in self.v])

    def __mul__(self, other):
        """Scalar or Vector multiplication"""
        # JIT Path
        if self.is_lazy or (isinstance(other, FractalVector7) and other.is_lazy):
             left = self._node if self.is_lazy else constant(self.v if isinstance(self.v, (int,float)) else 0.0)
             
             if isinstance(other, FractalVector7):
                 right = other._node
                 return FractalVector7(node=fractal_mul(left, right))
             elif isinstance(other, (int, float)):
                 right = constant(other)
                 return FractalVector7(node=algebraic_op('*', left, right))
                 
        if isinstance(other, (int, float, complex)):
            return FractalVector7([x * other for x in self.v])
        return NotImplemented

    def __rmul__(self, scalar):
        """Reverse scalar multiplication: c * v"""
        return self.__mul__(scalar)

    def apply_complex_sin(self):
        """Apply sin(z) element-wise"""
        return FractalVector7([cmath.sin(x) for x in self.v])


class FractalMatrix7:
    """
    A 7x7 matrix transformation.
    Stored as a list of 7 FractalVector7 (rows).
    """
    def __init__(self, rows=None):
        if rows is None:
            self.rows = [FractalVector7() for _ in range(7)]
        else:
            if len(rows) != 7:
                raise ValueError("FractalMatrix7 must have 7 rows")
            self.rows = rows

    @staticmethod
    def identity():
        rows = []
        for i in range(7):
            v = [0j] * 7
            v[i] = 1.0 + 0j
            rows.append(FractalVector7(v))
        return FractalMatrix7(rows)

    def dot(self, vec: FractalVector7) -> FractalVector7:
        """Matrix-Vector multiplication: y = M @ x"""
        result = []
        for row in self.rows:
            # Dot product of row and vec
            # sum(row[i] * vec[i])
            val = sum(r * x for r, x in zip(row.v, vec.v))
            result.append(val)
        return FractalVector7(result)

    def transpose(self):
        """Return transpose matrix"""
        new_rows = []
        for col_idx in range(7):
            col_vals = [self.rows[row_idx].v[col_idx] for row_idx in range(7)]
            new_rows.append(FractalVector7(col_vals))
        return FractalMatrix7(new_rows)
