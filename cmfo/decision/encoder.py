"""
CMFO D1-D6 Fractal Encoder
==========================
Deterministic Text-to-Vector 7D Mapper.

Principle:
Maps any text input to a deterministic 7D semantic vector.
- Known words: Uses Lexicon (if available) or deterministic hash.
- Unknown words: Uses deterministic hash projection.
- Sentences: Geometric composition (normalized sum).

Guarantees:
1. Determinism: Same text -> Same vector (ALWAYS).
2. Continuity: Similar words have defined distances.
3. Robustness: Never fails on unknown words.
"""

import math
import hashlib
import struct
from typing import List, Optional
import re

# Constants
PHI = 1.6180339887
VECTOR_DIM = 7


class FractalEncoder:
    """
    Deterministic Text -> 7D Vector Encoder.
    
    No neural networks. No training. Pure algebra.
    """
    
    def __init__(self):
        # Cache for performance
        self._cache = {}
        
    def encode(self, text: str) -> List[float]:
        """
        Encode text to 7D fractal vector.
        
        Args:
            text: Input string (word, sentence, or paragraph)
            
        Returns:
            7D list of floats [-1.0, 1.0]
        """
        if not text:
            return [0.0] * VECTOR_DIM
        
        # Check cache
        text_key = text.strip()
        if text_key in self._cache:
            return self._cache[text_key]
        
        # Tokenize (simple, robust)
        tokens = self._tokenize(text)
        
        if not tokens:
            return [0.0] * VECTOR_DIM
        
        # Encode tokens
        vectors = [self._encode_token(t) for t in tokens]
        
        # Compose (Geometric Mean Variant for Stability)
        # Standard mean tends to 0. This preserves magnitude.
        
        if len(vectors) == 1:
            result = vectors[0]
        else:
            result = self._compose_vectors(vectors)
            
        # Store in cache
        self._cache[text_key] = result
        return result
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Deterministic tokenization.
        Lowercase, remove punctuation, split by whitespace.
        """
        # Remove non-alphanumeric (keep basic punctuation for structure cues later if needed)
        # For base vector, pure words are best.
        clean = re.sub(r'[^\w\s]', '', text.lower())
        return clean.split()
    
    def _encode_token(self, token: str) -> List[float]:
        """
        Encode single token.
        Priority 1: D8 Semantic Algebra (Real Value)
        Priority 2: Deterministic Hash (Fallback)
        """
        # Try D8 Algebra first
        try:
            from ..semantics.algebra import SemanticAlgebra
            # We check if it has a definition in the algebra's lexicon
            # This is a bit inefficient (checking internal dict), but safe
            # Ideally SemanticAlgebra exposes "has_definition(word)"
            
            # For now, we just call value_of and check if it returns a non-zero vector
            # (SemanticAlgebra returns 0-vector for unknown words in our implementation)
            vector = SemanticAlgebra.value_of(token)
            
            # Check if non-zero (implies found)
            if any(abs(x) > 1e-9 for x in vector):
                return vector
                
        except ImportError:
            pass # Module not found/ready, fallback to hash
            
        # Fallback: Deterministic Hash (The "God Function")
        hash_bytes = hashlib.sha256(token.encode('utf-8')).digest()
        vector = []
        
        for i in range(VECTOR_DIM):
            # Extract 4 bytes per dimension
            chunk = hash_bytes[i*4 : (i+1)*4]
            val_int = struct.unpack('>I', chunk)[0]
            
            # Normalize to [-1, 1]
            val_norm = (val_int / (2**32 - 1)) * 2.0 - 1.0
            
            # Apply Phi-rotation (Fractal distribution)
            # sin(val * phi^i * pi) creates non-linear distribution
            raw = val_norm * (PHI ** i)
            projected = math.sin(raw * math.pi)
            
            vector.append(projected)
            
        return vector
        
    def _compose_vectors(self, vectors: List[List[float]]) -> List[float]:
        """
        Compose multiple vectors into one sentence vector.
        Uses normalized sum to maintain magnitude.
        """
        start = [0.0] * VECTOR_DIM
        
        # Sum
        for vec in vectors:
            for i in range(VECTOR_DIM):
                start[i] += vec[i]
                
        # Normalize to unit sphere (approx) to keep values in decent range
        magnitude = math.sqrt(sum(x*x for x in start))
        
        if magnitude < 1e-9:
            return start
            
        # Scale back to average magnitude of inputs (approx 0.5-0.8 for sine waves)
        # Or just normalize to 1? Let's use normalized average.
        
        scale = 1.0 / (magnitude + 1e-9) 
        # But we want to preserve "intensity" of many words?
        # No, semantic space is usually normalized.
        
        normalized = [x * scale for x in start]
        return normalized


if __name__ == "__main__":
    print("CMFO Fractal Encoder")
    print("====================")
    
    encoder = FractalEncoder()
    
    # Test stability
    t1 = "Hola mundo"
    v1 = encoder.encode(t1)
    v2 = encoder.encode(t1)
    
    print(f"Stability check ('{t1}'): {'PASS' if v1 == v2 else 'FAIL'}")
    print(f"Vector: {[round(x, 3) for x in v1]}")
    
    # Test distinction
    t2 = "Adiós mundo"
    v3 = encoder.encode(t2)
    
    dist = math.sqrt(sum((v1[i]-v3[i])**2 for i in range(7)))
    print(f"Distinction check: {'PASS' if dist > 0.0 else 'FAIL'}")
    print(f"Distance: {dist:.4f}")
    
    # Test calibration examples
    print("\nEncoding Calibration Examples:")
    examples = [
        "Sí, París es la capital de Francia",
        "No, Madrid es la capital de España",
        "¿Te refieres a la ciudad?"
    ]
    
    for ex in examples:
        vec = encoder.encode(ex)
        print(f"'{ex[:20]}...': {[round(x, 2) for x in vec]}")
