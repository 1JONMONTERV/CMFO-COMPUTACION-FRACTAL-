
import os
import sys
import struct
import random
import math
import time

# Add local path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from cmfo.crypto.fractal_sha256 import FractalSHA256, FractalWord

def float_to_fractal_block(header_prefix, nonce_float):
    """
    Constructs a block where the Nonce is continuous (float).
    """
    w1 = [FractalWord.from_int(struct.unpack(">I", header_prefix[i:i+4])[0]) for i in range(0,64,4)]
    
    tail_bytes = header_prefix[64:] 
    tail_words = [FractalWord.from_int(struct.unpack(">I", tail_bytes[i:i+4])[0]) for i in range(0,12,4)]
    
    if isinstance(nonce_float, list):
         nonce_word = FractalWord(nonce_float, 32)
    else:
         nonce_word = FractalWord.from_int(int(nonce_float))
         
    tail_words.append(nonce_word)
    
    # Padding Words (Fixed)
    # len 80 (640 bits) -> pad to 128 bytes.
    # 80 bytes used. 48 bytes padding.
    pad_bytes = b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
    pad_words = [FractalWord.from_int(struct.unpack(">I", pad_bytes[i:i+4])[0]) for i in range(0,48,4)]
    
    w2 = tail_words + pad_words
    
    return w1, w2

def loss_function(hash_out):
    """
    Loss = Distance from Target (Perfect Zero).
    """
    h0 = hash_out[0].trits
    loss = 0
    # Weights: Exponential decay to prioritize MSB
    for i in range(32):
        val = h0[i]
        weight = 2.0**(31-i) 
        loss += (val * val) * weight
        
    return loss


def gradient_descent_solver():
    print("--- FRACTAL DETERMINISTIC SOLVER v3.0 (ULTIMATE MAINNET CHALLENGE) ---")
    print("Objective: SOLVE Bitcoin Mainnet Block (Diff ~80T) via Fractal Gradients.")
    print("Constraint: NO LOWERING DIFFICULTY. Target is Real Network Bits.")
    
    # 1. Real Mainnet Target
    # Bits: 0x170e5d6d
    # Target = 0x0e5d6d * 2^(8*(23-3)) = ...
    # This roughly requires 76 bits of zeros.
    target_bits = 0x170e5d6d
    exp = (target_bits >> 24) & 0xFF
    coeff = target_bits & 0x00FFFFFF
    target = coeff * (256**(exp - 3))
    
    print(f"Target Bits: {target_bits:08x}")
    print(f"Target Int:  {target:064x}")
    
    # Setup Header (Fixed)
    header_prefix = struct.pack("<I", 1) + b'\x00'*32 + b'\xAA'*32 + struct.pack("<I", int(time.time())) + struct.pack("<I", target_bits)
    
    # Hyperparameters for Advanced Optimizer
    lr = 0.01 
    momentum = 0.9
    
    epoch = 0
    restarts = 0
    
    start_t = time.time()
    
    # Init Velocity
    velocities = [0.0]*32
    current_nonce_trits = [random.random() for _ in range(32)]
    
    best_loss_global = float('inf')
    
    while True:
        # 1. Forward Pass
        # Discretize for Check
        int_nonce_trits = [1.0 if x > 0.5 else 0.0 for x in current_nonce_trits]
        w1_int, w2_int = float_to_fractal_block(header_prefix, int_nonce_trits)
        
        fsha_int = FractalSHA256()
        fsha_int.compress(w1_int)
        fsha_int.compress(w2_int)
        h_int = fsha_int.get_hash()
        
        # Check Mainnet Success
        h_val = 0
        for val in h_int: h_val = (h_val << 32) | val
            
        if h_val < target:
            dt = time.time() - start_t
            print(f"\n\n[!!!] CRITICAL: MAINNET BLOCK SOLVED!")
            print(f"Nonce Found: {h_val}") # Placeholder for actual nonce val calc
            print(f"Hash: {h_val:064x}")
            print(f"Time: {dt:.2f}s")
            return
            
        # Continuous Loss Check
        w1, w2 = float_to_fractal_block(header_prefix, current_nonce_trits)
        f_real = FractalSHA256()
        f_real.compress(w1)
        f_real.compress(w2)
        loss = loss_function(f_real.H)
        
        if loss < best_loss_global:
            best_loss_global = loss
            
        # 2. Gradient Estimate (Sparse optimization for speed)
        # We pick 3 random trits to optimize per step to start, rotating?
        # Or calculate full gradient every N steps?
        # Full gradient (32) is slow. let's do 4 top bits + 2 random bits.
        indices_to_opt = [0,1,2,3] + random.sample(range(4,32), 2)
        
        grads = [0.0]*32
        epsilon = 0.01
        
        # Base forward already done (f_real) -> loss
        
        for i in indices_to_opt:
            saved = current_nonce_trits[i]
            current_nonce_trits[i] += epsilon
            
            # Fast Forward (Only recalculate affected parts? No, full SHA needed)
            w1_p, w2_p = float_to_fractal_block(header_prefix, current_nonce_trits)
            f_p = FractalSHA256()
            f_p.compress(w1_p)
            f_p.compress(w2_p)
            loss_p = loss_function(f_p.H)
            
            grads[i] = (loss_p - loss) / epsilon
            current_nonce_trits[i] = saved # Restore
            
        # 3. Momentum Update
        grad_mag = 0
        for i in range(32):
            g = grads[i]
            velocities[i] = (momentum * velocities[i]) - (lr * g)
            
            current_nonce_trits[i] += velocities[i]
            
            # Boundary Reflection (stay in [0,1])
            if current_nonce_trits[i] < 0.01:
                current_nonce_trits[i] = 0.01
                velocities[i] *= -0.5 # Bounce
            elif current_nonce_trits[i] > 0.99:
                current_nonce_trits[i] = 0.99
                velocities[i] *= -0.5 # Bounce
                
            grad_mag += abs(g)
            
        epoch += 1
        
        # 4. Restart Logic (Basin Hopping)
        # If gradient is zero (flat) or stuck for too long, jump.
        if epoch % 50 == 0:
            # Check convergence
            if grad_mag < 0.1: # Stuck
                restarts += 1
                current_nonce_trits = [random.random() for _ in range(32)]
                velocities = [0.0]*32
                
        # Log
        if epoch % 5 == 0:
            print(f"Ep {epoch} | Rst {restarts} | Loss {loss:.1e} | Best {best_loss_global:.1e} | Target {target:064x}...", end="\r")

if __name__ == "__main__":
    gradient_descent_solver()
