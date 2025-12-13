import sys

class PhiLogic:
    def __init__(self, phi=1.61803398875):
        self.phi = phi

    def and_op(self, a, b):
        return min(a, b) * (1.0 / self.phi)

    def or_op(self, a, b):
        # In this demo we allow values > 1 to show geometric growth
        return max(a, b) * self.phi

    def not_op(self, a):
        return self.phi - a

def print_truth_table(name, logic_system):
    print(f"\n--- {name} (phi={logic_system.phi:.4f}) ---")
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    
    print(f"{'A':<5} {'B':<5} | {'AND':<10} {'OR':<10}")
    print("-" * 35)
    
    for a, b in inputs:
        and_val = logic_system.and_op(a, b)
        or_val = logic_system.or_op(a, b)
        print(f"{a:<5} {b:<5} | {and_val:<10.4f} {or_val:<10.4f}")
    
    # Check NOT
    print(f"\n{'A':<5} | {'NOT':<10}")
    print("-" * 18)
    for a in [0, 1]:
        not_val = logic_system.not_op(a)
        print(f"{a:<5} | {not_val:<10.4f}")

def verify_boolean_limit():
    print("\n=== VERIFICATION: Boolean Degeneracy ===")
    
    # 1. Instantiate Logic with phi=1.0
    bool_logic = PhiLogic(phi=1.0)
    
    # 2. Check Standard Truth Table
    # Expected:
    # 0,0 -> AND 0, OR 0
    # 0,1 -> AND 0, OR 1
    # 1,1 -> AND 1, OR 1
    # NOT 0 -> 1, NOT 1 -> 0
    
    failures = 0
    
    # AND Check
    if bool_logic.and_op(1, 1) != 1.0: failures += 1
    if bool_logic.and_op(0, 1) != 0.0: failures += 1
    
    # OR Check
    if bool_logic.or_op(0, 0) != 0.0: failures += 1
    if bool_logic.or_op(1, 0) != 1.0: failures += 1
    
    # NOT Check
    if bool_logic.not_op(0) != 1.0: failures += 1
    if bool_logic.not_op(1) != 0.0: failures += 1
    
    if failures == 0:
        print("[SUCCESS] Phi-Logic with phi=1.0 behaves EXACTLY like Boolean Logic.")
    else:
        print(f"[FAIL] Discrepancies found: {failures} errors.")
        sys.exit(1)

if __name__ == "__main__":
    # Demo Fractal Logic
    fractal = PhiLogic(phi=1.61803398875)
    print_truth_table("Fractal Logic", fractal)
    
    # Demo Boolean Limit
    boolean = PhiLogic(phi=1.0)
    print_truth_table("Boolean Limit", boolean)
    
    verify_boolean_limit()
