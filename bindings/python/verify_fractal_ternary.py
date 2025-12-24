
import sys
import os

# Add local path to import cmfo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bindings/python')))

from cmfo.logic.phi_logic import fractal_and, fractal_or, fractal_xor

def print_table(op_name, op_func):
    print(f"\n--- {op_name} Truth Table ---")
    print(f"{'A':>4} | {'B':>4} | {'Result':>8}")
    print("-" * 22)
    # Check boolean subset {0, 1} first? Or full {-1, 0, 1}
    # User said "Reducen a booleano cuando x,y in {0,1}"
    # Let's check {-1, 0, 1} specifically.
    
    values = [-1, 0, 1]
    
    for a in values:
        for b in values:
            res = op_func(a, b)
            print(f"{a:>4} | {b:>4} | {res:8.4f}")

def check_boolean_reduction():
    print("\n--- Boolean Reduction Check {0, 1} ---")
    vals = [0, 1]
    
    # Standard Boolean Expectation
    # AND: 0&0=0, 0&1=0, 1&0=0, 1&1=1
    # OR:  0|0=0, 0|1=1, 1|0=1, 1|1=1
    # XOR: 0^0=0, 0^1=1, 1^0=1, 1^1=0
    
    print(f"{'Op':>4} | {'A':>2} | {'B':>2} | {'Fractal':>8} | {'Bool':>4} | {'Match'}")
    
    for a in vals:
        for b in vals:
            # AND
            f_and = fractal_and(a, b)
            b_and = 1 if (a and b) else 0
            match = abs(f_and - b_and) < 0.001
            print(f"AND  | {a:>2} | {b:>2} | {f_and:8.4f} | {b_and:>4} | {match}")
            
            # OR
            f_or = fractal_or(a, b)
            b_or = 1 if (a or b) else 0
            match = abs(f_or - b_or) < 0.001
            print(f"OR   | {a:>2} | {b:>2} | {f_or:8.4f} | {b_or:>4} | {match}")
            
            # XOR
            f_xor = fractal_xor(a, b)
            b_xor = 1 if (a != b) else 0
            match = abs(f_xor - b_xor) < 0.001
            print(f"XOR  | {a:>2} | {b:>2} | {f_xor:8.4f} | {b_xor:>4} | {match}")

if __name__ == "__main__":
    print("=== Balanced Ternary Fractal Logic Verification ===")
    print_table("AND Fractal", fractal_and)
    print_table("OR Fractal", fractal_or)
    print_table("XOR Fractal", fractal_xor)
    
    check_boolean_reduction()
