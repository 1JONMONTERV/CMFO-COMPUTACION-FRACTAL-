import sys
import os

sys.path.append(os.getcwd())
from d28_edu_eval.evaluator import SovereignEvaluator

def test_evaluation():
    print("[*] TEST: CMFO-EDU-EVAL (D28)")
    print("=" * 60)
    
    evaluator = SovereignEvaluator("d28_edu_eval/rubrics_math_10.json")
    
    # Case 1: Binomial Expansion (Correct)
    # Input: "(a+b)^2 = a^2 + 2ab + b^2"
    ans_1 = "a^2 + 2ab + b^2"
    print(f"\n[Problem]: Expand (a+b)^2")
    print(f"[Student]: {ans_1}")
    res_1 = evaluator.evaluate("binomial_expansion", ans_1)
    print(f"[CMFO]: {res_1['status']} | Grade: {res_1['grade']}")
    
    if res_1["status"] != "CORRECT":
        print("FAIL: Should be CORRECT")
        
    # Case 2: Binomial Expansion (Axiom Violation)
    # Input: "a^2 + b^2" (The Freshman's Dream error)
    ans_2 = "a^2 + b^2"
    print(f"\n[Problem]: Expand (a+b)^2")
    print(f"[Student]: {ans_2}")
    res_2 = evaluator.evaluate("binomial_expansion", ans_2)
    print(f"[CMFO]: {res_2['status']} | {res_2['violation_type']}")
    print(f"[Feedback]: {res_2['feedback']}")
    
    if res_2["status"] != "AXIOM_VIOLATION": 
         print(f"FAIL: Should be AXIOM_VIOLATION. Got {res_2['status']}")

    # Case 3: Linear Eq (Correct scalar)
    ans_3 = "x = 7"
    print(f"\n[Problem]: Solve 2x - 4 = 10")
    print(f"[Student]: {ans_3}")
    res_3 = evaluator.evaluate("linear_eq_solve", ans_3)
    print(f"[CMFO]: {res_3['status']} | Grade: {res_3['grade']}")

    # Case 4: Linear Eq (Sign Error)
    # 2x = 6 (subtracted 4 instead of adding) -> x=3
    ans_4 = "x = 3"
    print(f"\n[Problem]: Solve 2x - 4 = 10")
    print(f"[Student]: {ans_4}")
    res_4 = evaluator.evaluate("linear_eq_solve", ans_4)
    print(f"[CMFO]: {res_4['status']} | {res_4.get('violation_type', 'None')}")
    print(f"[Feedback]: {res_4.get('feedback')}")

if __name__ == "__main__":
    test_evaluation()
