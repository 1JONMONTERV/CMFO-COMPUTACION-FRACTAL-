"""
CMFO Geometry - Triangle
=========================

The Triangle is the unit minimal of determination in CMFO.
It replaces the concept of a 'bit' with a geometric structure that carries 
state, relation, and scale.

Definition:
    Triangle = (state, relation, scale)
             = (x, f(x), φ^k)
"""

from ..constants import PHI

class Triangle:
    """
    Fundamental unit of determination.
    
    A Triangle creates a deterministic decision by fixing orientation and scale.
    unlike a probabilistic bit, a Triangle cannot differ from itself.
    
    Attributes
    ----------
    state : any
        The base state (x).
    relation : any
        The relational state (f(x)).
    scale : float
         The fractal scale factor (φ^k).
    """
    
    def __init__(self, state, relation, scale=1.0):
        self.state = state
        self.relation = relation
        self.scale = scale
        
    def __repr__(self):
        return f"Triangle(x={self.state}, f(x)={self.relation}, scale={self.scale})"
    
    def decide(self):
        """
        Produce a deterministic decision.
        The triangle stabilizes the state via the relation.
        """
        # Placeholder for the "decision" logic logic if detailed elsewhere.
        # For v1.0 structure, identity or relation check is sufficient.
        return self.relation(self.state) if callable(self.relation) else self.relation
