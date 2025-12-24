import hmac
import hashlib
import json
import os
import sys
import struct
from typing import Dict, List, Optional, Tuple

sys.path.append(os.getcwd())
from cmfo.security.fractal_cipher import FractalCipher
from cmfo.semantics.algebra import SemanticAlgebra

class AuditLock:
    """
    CMFO-AUDIT-LOCK v1
    Hybrid Sovereign Security Protocol.
    
    Layer 1: Structural Derivation (FractalCipher)
             Ensures key exists only within valid context/structure.
    Layer 2: Standard Cryptography (HKDF + Stream Cipher)
             Ensures robustness matching industrial standards.
    """

    def __init__(self):
        self.fractal_engine = FractalCipher()

    def _derive_fractal_key(self, identity_vector: List[float], context_axioms: List[str]) -> bytes:
        """
        Step 1: Structural Derivation.
        Rotates the identity vector through the context axioms to produce K_fractal.
        """
        # "Subject" is the identity (Who is acting?)
        # "Context" is the transformation (Rotations based on axioms)
        
        current_vec = identity_vector
        
        for axiom in context_axioms:
            # We use semantic vector of the axiom as the rotation key
            axiom_vec = SemanticAlgebra.value_of(axiom)
            # Apply rotation (Unitary Transform)
            current_vec = self.fractal_engine.encrypt_vector(current_vec, axiom_vec)
        
        # Serialize the final 7D vector to bytes
        # Packing 7 floats (doubles) -> 56 bytes
        k_fractal = struct.pack(f'{self.fractal_engine.DIM}d', *current_vec)
        return k_fractal

    def _hkdf_derive(self, input_key_material: bytes, salt: bytes, info: bytes) -> bytes:
        """
        Step 2: HKDF (RFC 5869) using HMAC-SHA256.
        Standard derivation to whiten the fractal key into a crypto key.
        """
        # Extract
        prk = hmac.new(salt, input_key_material, hashlib.sha256).digest()
        
        # Expand
        # T(1) = HMAC(PRK, T(0) | info | 0x01)
        k_crypto = hmac.new(prk, info + b'\x01', hashlib.sha256).digest()
        return k_crypto

    def _standard_encrypt(self, key: bytes, plaintext: bytes) -> Tuple[bytes, bytes]:
        """
        Step 3: Standard Encryption.
        For this pilot (std lib only), we implement AES-CTR like stream cipher 
        using HMAC-SHA256 as a PRNG. 
        IN PRODUCTION: Replace with AES-GCM or ChaCha20-Poly1305.
        """
        nonce = os.urandom(16)
        
        # Generate enough keystream
        keystream = bytearray()
        block_counter = 0
        while len(keystream) < len(plaintext):
            counter_bytes = struct.pack('>Q', block_counter)
            block = hmac.new(key, nonce + counter_bytes, hashlib.sha256).digest()
            keystream.extend(block)
            block_counter += 1
            
        ciphertext = bytearray(len(plaintext))
        for i in range(len(plaintext)):
            ciphertext[i] = plaintext[i] ^ keystream[i]
            
        return nonce, bytes(ciphertext)

    def _standard_decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
        """
        Step 3 (Inverse): Standard Decryption.
        """
        # Keystream generation is identical
        keystream = bytearray()
        block_counter = 0
        while len(keystream) < len(ciphertext):
            counter_bytes = struct.pack('>Q', block_counter)
            block = hmac.new(key, nonce + counter_bytes, hashlib.sha256).digest()
            keystream.extend(block)
            block_counter += 1
            
        plaintext = bytearray(len(ciphertext))
        for i in range(len(ciphertext)):
            # XOR is its own inverse
            plaintext[i] = ciphertext[i] ^ keystream[i]
            
        return bytes(plaintext)

    def lock_log(self, identity: str, domain_context: List[str], log_entry: Dict) -> Dict:
        """
        Public API: Encrypts a log entry.
        """
        # 1. Fractal Layer
        id_vec = SemanticAlgebra.value_of(identity)
        k_fractal = self._derive_fractal_key(id_vec, domain_context)
        
        # 2. Derivation Layer
        # Salt could be the timestamp from log, but here we generate random salt
        salt = os.urandom(16) 
        k_crypto = self._hkdf_derive(k_fractal, salt, b"CMFO-AUDIT-LOCK")
        
        # 3. Encryption Layer
        payload_bytes = json.dumps(log_entry).encode('utf-8')
        nonce, ciphertext = self._standard_encrypt(k_crypto, payload_bytes)
        
        return {
            "locked": True,
            "version": "v1",
            "salt": salt.hex(),
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex(),
            "structural_context": domain_context # Needed to derive key to unlock
        }

    def unlock_log(self, identity: str, structural_context: List[str], locked_entry: Dict) -> Dict:
        """
        Public API: Unlock using Structural + Standard keys.
        """
        # 1. Reconstruct Fractal Key (Must match original context exactly)
        id_vec = SemanticAlgebra.value_of(identity)
        
        # Verify Context Match (Optional, or just try decrypting)
        # If contexts diverge, K_fractal will diverge -> K_crypto random -> Noise.
        
        k_fractal = self._derive_fractal_key(id_vec, structural_context)
        
        # 2. Derive Crypto Key
        salt = bytes.fromhex(locked_entry["salt"])
        k_crypto = self._hkdf_derive(k_fractal, salt, b"CMFO-AUDIT-LOCK")
        
        # 3. Decrypt
        nonce = bytes.fromhex(locked_entry["nonce"])
        ciphertext = bytes.fromhex(locked_entry["ciphertext"])
        
        try:
            plaintext_bytes = self._standard_decrypt(k_crypto, nonce, ciphertext)
            return json.loads(plaintext_bytes.decode('utf-8'))
        except Exception as e:
            return {"error": "Decryption Failed", "detail": "Structural Mismatch or Corruption"}

