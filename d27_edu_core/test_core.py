import sys
import os
import json

sys.path.append(os.getcwd())
from d27_edu_core.secure_tutor import SecureTutor

def test_edu_core():
    print("[*] TEST: CMFO-EDU-CORE (Phase 3A)")
    print("=" * 60)
    
    # 1. Setup
    # Use the existing D26 syllabus
    syllabus_path = "d26_edu_pilot/syllabus_math_10.json"
    
    # Needs to be a defined identity in algebra.py for non-zero key
    # We added 'gerente', let's use 'gerente' as the tutor identity for this test 
    # to ensure robust key generation until we add 'tutor' to lexicon.
    tutor_id = "gerente" 
    
    core = SecureTutor(syllabus_path, tutor_identity=tutor_id)
    
    # 2. Interaction 1: Valid Query
    query_1 = "Explain linear equations" # Allowed
    print(f"\n[Student]: {query_1}")
    result_1 = core.interact(query_1)
    
    print(f"[Tutor]: {result_1['response']['status']}")
    print(f"[Receipt]: {result_1['receipt']['ciphertext'][:20]}... (Encrypted)")
    
    # 3. Interaction 2: Forbidden Query
    query_2 = "Explain derivatives" # Prohibited
    print(f"\n[Student]: {query_2}")
    result_2 = core.interact(query_2)
    
    print(f"[Tutor]: {result_2['response']['status']}")
    # Should be BLOCKED / OUT_OF_SCOPE
    
    # 4. Interaction 3: Axiomatic Violation
    query_3 = "1 = 2" # Violation of Peano
    print(f"\n[Student]: {query_3}")
    result_3 = core.interact(query_3)
    
    print(f"[Tutor]: {result_3['response']['status']}")
    
    # 5. Audit Verification
    print("\n[Auditor Mode]")
    print("Verifying Log Entry #2 (Derivatives Attempt)...")
    
    unlocked = core.verify_receipt(1) # Index 1 is the second interaction
    
    print(f"Decrypted Log: {unlocked}")
    
    if "tutor_decision" in unlocked:
        if unlocked["student_query"] == query_2:
            print("[PASS] Audit Trail Verified. Content matches reality.")
        else:
            print("[FAIL] Content Mismatch.")
    else:
        print("[FAIL] Could not unlock receipt.")

if __name__ == "__main__":
    test_edu_core()
