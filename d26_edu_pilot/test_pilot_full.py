import sys
import os
import json

# Ensure imports work
sys.path.append(os.getcwd())

from d26_edu_pilot.tutor import PedagogicalTutor

def run_pilot_simulation():
    print("[*] CMFO-EDU-MATH-10: Starting Pilot Simulation")
    print("=" * 60)

    # Initialize Tutor with official syllabus
    tutor = PedagogicalTutor("d26_edu_pilot/syllabus_math_10.json")

    scenarios = [
        {
            "desc": "Forbidden Concept (Calculus)",
            "query": "Quiero calcular la derivada de x^2"
        },
        {
            "desc": "Allowed Concept (Analytic Geometry)",
            "query": "Explícame qué es la pendiente de una recta"
        },
        {
            "desc": "Mathematical Execution (Solving)",
            "query": "Ayúdame con 2x + 4 = 10"
        },
        {
            "desc": "Unknown Concept (No Proof in KB)",
            "query": "Dime quién ganó el mundial" # Matches no math concept, likely validated as UNKNOWN or passed validator but fails Proof check?
            # Actually weak validation might pass it if not forbidden, but Gate will fail due to no Proof.
        }
    ]

    for s in scenarios:
        print(f"\nScenario: {s['desc']}")
        print(f"Student: '{s['query']}'")
        
        response = tutor.process_student_query(s['query'])
        
        if response["type"] == "response":
            print(f"CMFO: {response['content']}")
            print(f"Receipt: GENERATED [Hash: {json.loads(response['receipt'])['hash'][:8]}...]")
            # print(response['receipt']) # Uncomment for full debug
        
        elif response["type"] == "block":
            print(f"CMFO [BLOCKED]: {response['message']}")
            print(f"Authority: {response['authority']}")
            
        elif response["type"] == "error":
             print(f"CMFO [ERROR]: {response['message']}")
             if "receipt" in response:
                 print(f"Receipt: BLOCKED [Reason: {json.loads(response['receipt'])['blocked_by']}]")

    print("\n" + "=" * 60)
    print("[*] Simulation Complete.")

if __name__ == "__main__":
    run_pilot_simulation()
