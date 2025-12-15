"""
CMFO Math Utilities - Pure Python
==================================

Replacement for NumPy operations using only Python standard library.
"""

import math
from typing import List, Union


def _norm(v: List[float]) -> float:
    """Calculate Euclidean norm (L2 norm) of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _dot(a: List[float], b: List[float]) -> float:
    """Calculate dot product of two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vectors must have same length: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def _sum(arr: List[float]) -> float:
    """Sum all elements in array."""
    return sum(arr)


def _mean(arr: List[float]) -> float:
    """Calculate mean of array."""
    return sum(arr) / len(arr) if arr else 0.0


def _linspace(start: float, stop: float, num: int) -> List[float]:
    """Generate evenly spaced numbers over interval."""
    if num <= 0:
        return []
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _zeros(shape: Union[int, tuple]) -> Union[List[float], List[List[float]]]:
    """Create array filled with zeros."""
    if isinstance(shape, int):
        return [0.0] * shape
    elif isinstance(shape, tuple) and len(shape) == 2:
        rows, cols = shape
        return [[0.0] * cols for _ in range(rows)]
    else:
        raise ValueError(f"Unsupported shape: {shape}")


def _zeros_like(arr: List) -> List:
    """Create array of zeros with same shape as input."""
    if isinstance(arr[0], list):
        return [[0.0] * len(row) for row in arr]
    return [0.0] * len(arr)


def _argmax(arr: List[float]) -> int:
    """Return index of maximum value."""
    if not arr:
        raise ValueError("Cannot find argmax of empty array")
    return max(range(len(arr)), key=lambda i: arr[i])


def _array(data, dtype=None) -> List:
    """Convert data to list (NumPy array replacement)."""
    if isinstance(data, list):
        return data
    elif isinstance(data, tuple):
        return list(data)
    elif hasattr(data, '__iter__'):
        return list(data)
    else:
        return [data]


__all__ = [
    '_norm', '_dot', '_sum', '_mean', '_linspace',
    '_zeros', '_zeros_like', '_argmax', '_array'
]
