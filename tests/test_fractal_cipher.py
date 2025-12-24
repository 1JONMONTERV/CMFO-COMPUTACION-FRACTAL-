import sys
import os
import math

sys.path.append(os.getcwd())
from cmfo.security.fractal_cipher import FractalCipher
from cmfo.semantics.algebra import SemanticAlgebra

def test_encryption():
    print("[*] CMFO Fractal Encryption (CFE) Test")
    print("=" * 50)
    
    cipher = FractalCipher()
    
    # 1. Define Key (Structural)
    # Key is the semantic vector of "verdad" (Truth)
    key_word = "verdad"
    key_vec = SemanticAlgebra.value_of(key_word)
    print(f"Key: '{key_word}' (Vector: {key_vec[:3]}...)")

    # 2. Define Payload
    message = "CMFO-SECURE-DATA"
    print(f"Original Message: {message}")
    
    # 3. Encrypt
    input_vectors = cipher.string_to_vector_stream(message)
    encrypted_vectors = [cipher.encrypt_vector(v, key_vec) for v in input_vectors]
    
    print(f"Encrypted (First Block): {[round(x,2) for x in encrypted_vectors[0]]}")
    
    # 4. Decrypt (Success Case)
    decrypted_vectors = [cipher.decrypt_vector(v, key_vec) for v in encrypted_vectors]
    restored_message = cipher.vector_stream_to_string(decrypted_vectors)
    print(f"Restored Message: {restored_message}")
    
    if restored_message == message:
        print("[SUCCESS] Reversible Logic Verified.")
    else:
        print("[FAIL] Decryption mismatch.")

    # 5. Decrypt (Failure Case - Structural Slight Mismatch)
    # Perturb key by epsilon ( simulating lost context time)
    noisy_key = [x + 0.01 for x in key_vec] 
    bad_vectors = [cipher.decrypt_vector(v, noisy_key) for v in encrypted_vectors]
    garbage_msg = cipher.vector_stream_to_string(bad_vectors)
    
    print("\nAttempting Decrypt with Noisy Key (Epsilon Error):")
    print(f"Result: '{garbage_msg}'")
    
    if garbage_msg != message:
        print("[SUCCESS] Structural Lock Verified (Noise produced garbage).")

if __name__ == "__main__":
    test_encryption()
