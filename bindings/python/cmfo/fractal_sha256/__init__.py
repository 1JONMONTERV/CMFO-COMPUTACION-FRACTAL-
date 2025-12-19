"""
CMFO - Fractal SHA-256d Implementation
=======================================

A bit-exact, reversible, and traceable implementation of SHA-256d
using fractal operators.

Conforms to Bitcoin specification while providing internal observability.
"""

from .fractal_state import FractalState, FractalCell
from .reversible_ops import (
    xor_fractal, and_fractal, not_fractal, 
    rotr_fractal, shr_fractal, add_mod_fractal
)
from .sha256_functions import (
    Ch_fractal, Maj_fractal, 
    Sigma0_fractal, Sigma1_fractal,
    sigma0_fractal, sigma1_fractal
)
from .sha256_engine import SHA256Fractal
from .sha256d import sha256d_fractal
