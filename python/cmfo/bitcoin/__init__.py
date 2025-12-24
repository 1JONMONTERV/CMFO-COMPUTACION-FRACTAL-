"""
Módulo Bitcoin: Inversión estructural del nonce
"""

from .byte_constraint_graph import (
    ByteConstraintGraph,
    ByteNode,
    ByteDomain,
    ByteConstraint,
    FixedValueConstraint,
    RangeConstraint,
    XORConstraint,
    ADDConstraint
)

from .nonce_restrictor import (
    NonceRestrictor,
    analyze_block,
    build_header,
    parse_header
)

__all__ = [
    'ByteConstraintGraph',
    'ByteNode',
    'ByteDomain',
    'ByteConstraint',
    'FixedValueConstraint',
    'RangeConstraint',
    'XORConstraint',
    'ADDConstraint',
    'NonceRestrictor',
    'analyze_block',
    'build_header',
    'parse_header'
]
