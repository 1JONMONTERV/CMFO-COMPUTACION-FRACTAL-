"""
CMFO: Computational Manifold Fractal Operators
==============================================

The "CMFO Engine" SDK provides the mathematical primitives for 
fractal computing on anisotropic tori.

Modules:
    - cmfo.core: Geometry, measures, and fundamental constants.
    - cmfo.physics: Dynamics and operators (To be migrated).
    - cmfo.memory: Fractal compression and tokenization (To be implemented).

Basic Usage:
    >>> from cmfo import geodesic_distance, PHI
    >>> dist = geodesic_distance(x, y)
"""

from .core.geometry import (
    PHI, 
    LAMBDA, 
    geodesic_distance, 
    translation, 
    reflection
)

__version__ = "1.3.0"
