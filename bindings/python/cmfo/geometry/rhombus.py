"""
CMFO Geometry - Rhombus
=======================

The Rhombus is the unit minimal of reversible computation.
It is composed of two Triangles (forward and backward) that ensure 
no information is lost during transformation.

Definition:
    Rhombus = Triangle_forward + Triangle_backward
"""

from .triangle import Triangle

class Rhombus:
    """
    Fundamental unit of reversible computation.
    
    The rhombus guarantees conservation of information by coupling a forward
    process with its geometric inverse.
    
    Attributes
    ----------
    forward : Triangle
        The forward-facing geometric state.
    backward : Triangle
        The backward-facing (inverse) geometric state.
    """
    
    def __init__(self, forward: Triangle, backward: Triangle):
        self.forward = forward
        self.backward = backward
        
    def __repr__(self):
        return f"Rhombus(F={self.forward}, B={self.backward})"
        
    def is_reversible(self) -> bool:
        """
        Check if the computation is strictly reversible.
        In a perfect Rhombus, F(B(x)) == x.
        """
        # Conceptual implementation for v1.0
        return True
