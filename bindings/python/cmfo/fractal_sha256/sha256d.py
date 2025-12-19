"""
SHA-256d Fractal Wrapper
=========================

Double SHA-256 implementation matching Bitcoin's hashing algorithm.
"""

from .sha256_engine import SHA256Fractal

def sha256d_fractal(message: bytes) -> bytes:
    """
    Compute SHA-256d (double SHA-256) using fractal implementation.
    
    Hash = SHA256(SHA256(message))
    
    Args:
        message: Input bytes (e.g., 80-byte block header)
        
    Returns:
        32-byte digest
    """
    engine = SHA256Fractal()
    
    # First pass
    digest1 = engine.hash(message)
    
    # Second pass
    digest2 = engine.hash(digest1)
    
    return digest2
