import sys
import os
import json

sys.path.append(os.getcwd())
try:
    from cmfo.security.audit_lock import AuditLock
except ImportError:
    # If package structure issues, direct import for test
    pass

def test_audit_lock():
    print("[*] CMFO-AUDIT-LOCK v1: Hybrid Sovereign Security Test")
    print("=" * 60)
    
    locker = AuditLock()
    
    # 1. Valid Context
    identity = "gerente" # Identity Vector
    context = ["verdad", "orden", "entidad"] # Structural Chain
    
    log_data = {
        "action": "AUTHORIZE_PAYMENT",
        "amount": 250000,
        "beneficiary": "BLACK_OPS_LLC"
    }
    
    print(f"Original Log: {json.dumps(log_data)}")
    print(f"Context: {context} | Identity: {identity}")
    
    # 2. Lock (Encrypt)
    locked_blob = locker.lock_log(identity, context, log_data)
    print(f"\nLocked Entry [Hybrid]:")
    print(f"  Ciphertext (Hex): {locked_blob['ciphertext'][:32]}...")
    print(f"  Salt: {locked_blob['salt']}")
    
    # 3. Unlock (Success)
    print("\nAttempting Unlock (Correct Structure)...")
    unlocked_data = locker.unlock_log(identity, context, locked_blob)
    print(f"Result: {unlocked_data}")
    
    if unlocked_data == log_data:
        print("[SUCCESS] Integrity Verified.")
    else:
        print("[FAIL] Integrity Check Failed.")
        
    # 4. Unlock (Failure - Structural Mismatch)
    # Attacker has the file (ciphertext, salt, nonce)
    # Attacker tries to unlock using 'hacker' identity or wrong context 'caos'
    
    print("\nAttempting Unlock (Wrong Structure - 'hacker' identity)...")
    hacker_identity = "hacker" # 'hacker' semantically != 'gerente'
    
    # Note: 'hacker' might not be in lexicon, falls back to zero vector -> K_fractal differs
    bad_unlock = locker.unlock_log(hacker_identity, context, locked_blob)
    
    # Since it's a stream cipher, it won't throw an error, it will produce garbage JSON which fails parsing
    # Our code catches json decode error and returns error dict
    print(f"Result: {bad_unlock}")
    
    if "error" in bad_unlock:
        print("[SUCCESS] Structural Defense Active (Decryption rejected).")
    else:
        # If it returns garbage data (unlikely to match but possible to decode as string? actually we check json)
        print(f"[FAIL] Should have failed. Got: {bad_unlock}")

    # 5. Unlock (Failure - Context Mismatch)
    print("\nAttempting Unlock (Correct ID, Wrong Context Axioms)...")
    wrong_context = ["mentira", "caos", "entidad"] # Twisted context
    bad_unlock_ctx = locker.unlock_log(identity, wrong_context, locked_blob)
    print(f"Result: {bad_unlock_ctx}")
    
    if "error" in bad_unlock_ctx:
        print("[SUCCESS] Contextual Defense Active.")

if __name__ == "__main__":
    test_audit_lock()
