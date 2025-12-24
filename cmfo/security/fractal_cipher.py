import math
import random
from typing import List
import sys
import os

# Link to core algebra (assuming run from root)
sys.path.append(os.getcwd())
# We need SemanticAlgebra. Let's import structurally or duplicate logic if simpler for prototype.
# Using relative import if possible, else minimal reimplementation of Vector ops.
try:
    from cmfo.semantics.algebra import SemanticAlgebra
except ImportError:
    # Minimal Vector Ops for standalone safety
    class SemanticAlgebra:
        @staticmethod
        def normalize(v):
            norm = math.sqrt(sum(x*x for x in v))
            if norm < 1e-9: return v
            return [x/norm for x in v]

class FractalCipher:
    """
    CMFO Fractal Encryption (CFE) Engine.
    "Inverse Boolean" Logic = 7D Unitary Rotation.
    
    Data is not XORed. It is Rotated.
    Symmetry: R * R^T = I
    """

    DIM = 7

    def __init__(self):
        pass

    def _generate_rotation_matrix(self, key_vector: List[float]) -> List[List[float]]:
        """
        Generates a 7x7 Unitary Rotation Matrix from a 7D key vector.
        Uses Householder reflection or simplified Gram-Schmidt to build an orthogonal basis
        where the key_vector is the primary axis.
        """
        # 1. Normalize Key (Axis 0)
        v0 = SemanticAlgebra.normalize(key_vector)
        basis = [v0]

        # 2. Complete the Basis (Gram-Schmidt)
        # We need self.DIM vectors total.
        # Iterate through standard basis vectors e_0 ... e_6
        # If a candidate is linearly independent, add it.
        
        for i in range(self.DIM):
            if len(basis) == self.DIM:
                break
                
            # Create a candidate vector (e_i)
            candidate = [0.0] * self.DIM
            candidate[i] = 1.0
            
            # Project out existing basis vectors (Gram-Schmidt)
            # v_new = v - sum( proj_u(v) )
            temp = list(candidate)
            for b_vec in basis:
                dot = sum(c * b for c, b in zip(temp, b_vec))
                temp = [c - dot * b for c, b in zip(temp, b_vec)]
            
            # Check if what remains is significant (not linearly dependent)
            norm = math.sqrt(sum(x*x for x in temp))
            if norm > 1e-5:
                # Normalize and add
                basis.append([x/norm for x in temp])
            
        # basis is a list of rows. This forms an Orthogonal Matrix Q.
        return basis

    def encrypt_vector(self, data_vector: List[float], key_vector: List[float]) -> List[float]:
        """
        R(key) * data -> cipher
        """
        # Ensure input is 7D (pad if needed, for prototype we assume 7D)
        if len(data_vector) != self.DIM:
            raise ValueError(f"Data must be {self.DIM}D vector")

        R = self._generate_rotation_matrix(key_vector)
        
        # Matrix-Vector Multiplication
        cipher = [0.0] * self.DIM
        for i in range(self.DIM):
            row = R[i]
            val = sum(row[j] * data_vector[j] for j in range(self.DIM))
            cipher[i] = val
            
        return cipher

    def decrypt_vector(self, cipher_vector: List[float], key_vector: List[float]) -> List[float]:
        """
        R(key)^T * cipher -> data
        (Inverse Boolean Operation: R^T is the inverse logic)
        """
        R = self._generate_rotation_matrix(key_vector)
        
        # Transpose Multiplication (Inverse)
        data = [0.0] * self.DIM
        for i in range(self.DIM):
            # Column j of R becomes Row j of R^T
            val = sum(R[row_idx][i] * cipher_vector[row_idx] for row_idx in range(self.DIM))
            data[i] = val
            
        return data

    def string_to_vector_stream(self, text: str) -> List[List[float]]:
        """Simple ASCII -> 7D chunks mapping for demo"""
        # Each char is a dimension? Or pack 7 chars?
        # Let's pack 7 chars per vector.
        bytes_data = text.encode('utf-8')
        vectors = []
        chunk = []
        for b in bytes_data:
            chunk.append(float(b))
            if len(chunk) == self.DIM:
                vectors.append(chunk)
                chunk = []
        if chunk:
            while len(chunk) < self.DIM: chunk.append(0.0)
            vectors.append(chunk)
        return vectors

    def vector_stream_to_string(self, vectors: List[List[float]]) -> str:
        out_bytes = bytearray()
        for v in vectors:
            for val in v:
                # Round to nearest int
                i = int(round(val))
                if i > 0: out_bytes.append(i) # 0 is padding
        return out_bytes.decode('utf-8', errors='ignore')

