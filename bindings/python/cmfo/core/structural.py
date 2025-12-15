"""
CMFO Structural Algebra
=======================
Pure Python implementation of 7-Dimensional Geometric Algebra.
Zero external dependencies (No NumPy).
"""

import math
import cmath

class FractalVector7:
    """
    A 7-dimensional vector in the fractal manifold.
    Stored as a list of 7 complex numbers.
    """
    __slots__ = ['v']

    def __init__(self, values=None):
        if values is None:
            self.v = [0j] * 7
        else:
            if len(values) != 7:
                raise ValueError("FractalVector7 must have exactly 7 components")
            self.v = list(values)

    def __repr__(self):
        # Pretty print resembling numpy
        formatted = ", ".join(f"{x:.2f}" for x in self.v)
        return f"FractalVector7([{formatted}])"

    def __add__(self, other):
        return FractalVector7([a + b for a, b in zip(self.v, other.v)])

    def __sub__(self, other):
        return FractalVector7([a - b for a, b in zip(self.v, other.v)])

    def norm(self):
        """Euclidean norm (L2)"""
        # sqrt( sum( |x|^2 ) )
        sum_sq = sum(abs(x)**2 for x in self.v)
        return math.sqrt(sum_sq)

    def normalize(self):
        """Return a normalized copy of the vector"""
        m = self.norm()
        if m < 1e-15:
            return FractalVector7(self.v) # Return zero vec if too small
        inv_m = 1.0 / m
        return FractalVector7([x * inv_m for x in self.v])

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
