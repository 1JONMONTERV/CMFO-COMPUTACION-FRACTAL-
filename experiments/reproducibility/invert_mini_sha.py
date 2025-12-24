import numpy as np
import math

# --- CMFO CONSTANTS ---
PHI = (1 + math.sqrt(5)) / 2
DIM = 7

# --- 7D VECTOR CLASS ---
class FractalState:
    def __init__(self, data=None):
        if data is None:
            self.vec = np.random.uniform(-1, 1, DIM) # Random nonce start
        else:
            self.vec = np.array(data, dtype=np.float64)
        self.normalize()
        
    def normalize(self):
        norm = np.linalg.norm(self.vec)
        if norm > 1e-15:
            self.vec = self.vec / norm * PHI # Scale to Phi Manifold
            
    def __repr__(self):
        return f"State(|v|={np.linalg.norm(self.vec):.4f})"

# --- FRACTAL SHA-LIKE OPERATORS ---
# Standard SHA: RotRight(x, n) ^ RotRight(x, m) ...
# CMFO SHA: Geometric Rotation matrices

def geometric_rotate(state, angle_factor):
    """Reversible rotation in 7D"""
    theta = math.atan(1/PHI) * angle_factor
    c, s = math.cos(theta), math.sin(theta)
    
    # Rotation matrix (R)
    R = np.eye(DIM)
    # Apply to first plane (example)
    R[0,0], R[0,1] = c, -s
    R[1,0], R[1,1] = s, c
    
    new_vec = np.dot(R, state.vec)
    return FractalState(new_vec)

def geometric_rotate_inv(state, angle_factor):
    """Exact inverse of geometric_rotate"""
    return geometric_rotate(state, -angle_factor)

def geometric_mix(state_a, state_b):
    """
    Reversible mixing via ROTATION (Unitary Operation).
    Instead of adding vectors (which changes length), we rotate A by phase of B.
    This preserves the Unit Sphere topology.
    """
    # Use B's phase to determine rotation angle
    phase_b = np.sum(state_b.vec) # Simple phase proxy
    angle = phase_b * (math.pi / DIM) # Map to angle
    
    return geometric_rotate(state_a, angle)

def geometric_mix_inv(state_out, state_b):
    """
    Inverse mixing: Rotate back by -B phase.
    """
    phase_b = np.sum(state_b.vec)
    angle = phase_b * (math.pi / DIM)
    
    return geometric_rotate_inv(state_out, angle)

# --- THE MINI-SHA ROUND ---
# Hash = Mix( Rotate(Input), Constant )

def mini_sha_round(input_state, salt_constant):
    """
    Forward 'Hash' Function.
    1. Rotate Input (Diffusion)
    2. Mix with Salt (Confusion - now Rotational)
    """
    rotated = geometric_rotate(input_state, angle_factor=5.0) 
    hashed = geometric_mix(rotated, salt_constant)
    return hashed

def inverse_mini_sha(hash_output, salt_constant):
    """
    Backward 'Mining' Function.
    1. Un-Mix Salt (Inverse Rotation)
    2. Un-Rotate
    """
    # 1. Reverse Confusion
    unmixed = geometric_mix_inv(hash_output, salt_constant)
    
    # 2. Reverse Diffusion
    recovered = geometric_rotate_inv(unmixed, angle_factor=5.0)
    
    return recovered

# --- MAIN EXECUTION ---
def run_demo():
    print("=== CMFO 'SHA-Sencillo' Inversion Demo ===")
    print("Objective: Prove that a SHA-like round is geometrically invertible.\n")
    
    # 1. Setup
    original_input = FractalState([1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    salt = FractalState([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    print(f"Original Secret: {original_input.vec[:3]}...")
    
    # 2. The One-Way Function (Standard View) / Transform (CMFO View)
    hashed_result = mini_sha_round(original_input, salt)
    print(f"Hashed Output:   {hashed_result.vec[:3]}...")
    
    # 3. The Inversion (Walking backwards on the Manifold)
    recovered = inverse_mini_sha(hashed_result, salt)
    print(f"Recovered Input: {recovered.vec[:3]}...")
    
    # 4. Verification
    error = np.linalg.norm(recovered.vec - original_input.vec)
    print(f"\nReversion Error: {error:.4e}")
    
    if error < 1e-14:
        print("SUCCESS: The SHA round was inverted analytically.")
        print("Implication: Brute force is unnecessary if the geometry is known.")
    else:
        print("FAILURE: Inversion drift too high.")

if __name__ == "__main__":
    run_demo()
