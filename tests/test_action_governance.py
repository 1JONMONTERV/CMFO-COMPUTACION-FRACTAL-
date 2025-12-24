import sys
import os

# Force ISO-8859-1 or UTF-8 if possible, and add root to path
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

sys.path.append(os.getcwd())

from cmfo.actions.gate import ActionGate

def test_governance():
    gate = ActionGate()
    print("[*] Testing CMFO Action Governance Matrix\n")

    scenarios = [
        {
            "name": "Trusted Calculation",
            "action": "execute_code",
            "domain": "MATH-10",
            "user_auth": True,
            "domain_auth": False, # Math domain doesn't explicitly authorize arbitrary code
            "proof_ref": "proof_123" # But we have a proof
        },
        {
            "name": "Malicious Code Injection",
            "action": "execute_code",
            "domain": "MATH-10",
            "user_auth": True,
            "domain_auth": False,
            "proof_ref": None # No proof
        },
        {
            "name": "Forbidden Derivation (10th Grade)",
            "action": "derive_concept",
            "domain": "CMFO-EDU-MATH-10",
            "user_auth": True,
            "domain_auth": False, # Syllabus said NO
            "proof_ref": "calculus_theorem_1" # Even if strictly true math-wise
        },
        {
            "name": "Medical Diagnosis",
            "action": "medical_diagnosis",
            "domain": "MEDICINE",
            "user_auth": True,
            "domain_auth": True, 
            "proof_ref": "symptom_match"
        }
    ]

    for s in scenarios:
        print(f"Scenario: {s['name']}")
        receipt = gate.attempt_execution(
            action=s['action'],
            domain=s['domain'],
            input_data="test_input",
            user_auth=s['user_auth'],
            domain_auth=s['domain_auth'],
            proof_ref=s['proof_ref']
        )
        print(receipt.to_json())
        print("-" * 60)

if __name__ == "__main__":
    test_governance()
