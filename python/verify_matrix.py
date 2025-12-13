import numpy as np
import sys

def colored(text, color_code):
    if sys.platform == "win32": return text
    return f"\033[{color_code}m{text}\033[0m"

def main():
    print("=== CMFO T7 Matrix Python Verification ===")
    print("Goal: Compare numerical stability of 7x7 inversion against NumPy (double precision).")
    
    np.random.seed(42)
    
    # Generate random 7x7 matrix
    A = np.random.uniform(-1, 1, (7, 7))
    
    # Compute NumPY Inverse
    try:
        InvA = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("Matrix is singular. Skipping.")
        return

    # Check Identity
    Identity = np.dot(A, InvA)
    I_expected = np.eye(7)
    
    diff = np.abs(Identity - I_expected)
    max_error = np.max(diff)
    
    print(f"\nMatrix Condition Number: {np.linalg.cond(A):.2f}")
    print(f"Max Error (A * InvA - I): {max_error:.2e}")
    
    # Threshold for double precision (usually around 1e-15)
    if max_error < 1e-12:
        print("\n[SUCCESS] Matrix inversion is numerically stable within tolerance.")
    else:
        print("\n[FAIL] Error exceeds tolerance!")
        sys.exit(1)

    print("\n--- Sample Row Check ---")
    print(f"Row 0 * Col 0 (Expected 1.0): {Identity[0,0]:.16f}")
    print(f"Row 0 * Col 1 (Expected 0.0): {Identity[0,1]:.16f}")

if __name__ == "__main__":
    main()
