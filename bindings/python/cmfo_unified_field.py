
import os
import sys
import struct
import math
import numpy as np
import random
import binascii

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from cmfo.crypto.fractal_sha256 import FractalSHA256, FractalWord

class UnifiedPhysics:
    """
    Models the SHA-256 State as a Unified Physical Field.
    Dimensions: Mass, Wave, Particle, Curve, Spiral.
    """
    
    def __init__(self):
        self.gravity_const = 6.674e-11
        
    def get_state_vector(self, header, nonce):
        # 1. Capture State (8 Words)
        h = bytearray(header)
        h[76:80] = struct.pack("<I", nonce)
        h_bytes = bytes(h)
        input_block = h_bytes + b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
        
        # Simplified Checksum/Hash State Logic for prototype
        # Re-using simple H2 generation from previous script logic 
        # But we need raw words to model "Physics"
        
        # We will use the standard INITIAL HASH as "Space"
        # and the compressed block as "Matter"
        # The result is the interaction.
        
        # Run SHA-256 Step to get final H
        chunk2 = input_block[64:]
        w = [FractalWord.from_int(struct.unpack(">I", chunk2[i:i+4])[0]) for i in range(0,64,4)]
        
        fsha = FractalSHA256() # Resets context
        # We need to feed it the MID-STATE from chunk 1 (approx)
        # For demo, we just model the nonce-chunk interaction
        fsha.compress(w)
        return np.array([float(x) for x in fsha.get_hash()], dtype=float)

    # --- DIMENSION 1: MASS (Gravitational Collapse) ---
    def metric_mass(self, state):
        # Model: The state collapses towards 0 (Black Hole).
        # Mass = Sum of bits (Energy).
        # Gravity = M1*M2 / r^2 ? 
        # Here: Density of the hash.
        total_mass = np.sum(state)
        # Schwarzschild Radius of the info?
        # Just normalized density.
        return total_mass / (8 * 2**32)

    # --- DIMENSION 2: WAVE (Spectral Resonance) ---
    def metric_wave(self, state):
        # FFT of the 8 words
        spectrum = np.abs(np.fft.fft(state))
        # Dominant Frequency Power
        return np.max(spectrum) / np.sum(spectrum)

    # --- DIMENSION 3: PARTICLE (Position/Momentum) ---
    def metric_particle(self, state):
        # Center of Mass of the 8-vector
        indices = np.arange(8)
        com = np.sum(indices * state) / np.sum(state)
        # Deviation from center (3.5)
        return abs(com - 3.5)

    # --- DIMENSION 4: CURVE (Geodesic Curvature) ---
    def metric_curve(self, state):
        # Second derivative of the sequence
        d1 = np.diff(state)
        d2 = np.diff(d1)
        # Smoothness energy
        energy = np.sum(d2**2)
        # Normalized by scale
        return 1.0 / (1.0 + energy / 1e18)

    # --- DIMENSION 5: SPIRAL (Vortex/Winding) ---
    def metric_spiral(self, state):
        # Phase angles
        angles = (state / 2**32) * 2 * np.pi
        # Sum of sine phases (constructive interference)
        # Vortex strength
        x = np.sum(np.cos(angles))
        y = np.sum(np.sin(angles))
        r = np.sqrt(x**2 + y**2)
        return r / 8.0 # Coherence 0..1

    def compute_super_metric(self, header, nonce):
        s = self.get_state_vector(header, nonce)
        
        m_mass = self.metric_mass(s)
        m_wave = self.metric_wave(s)
        m_part = self.metric_particle(s)
        m_curv = self.metric_curve(s)
        m_spir = self.metric_spiral(s)
        
        # The "Unified Field Score"
        # We assume Valid Blocks maximize ALL structure (Low Entropy, High Order)
        # So we want specific values.
        # But wait, Hash is random-looking.
        # Maybe Valid Block = MAXIMUM ENTROPY? (Perfect thermalization)
        # Or MAXIMUM SINGULARITY? (Zeroes)
        
        # User theory: "Attractor". Distinctive signature.
        # We return the vector.
        return np.array([m_mass, m_wave, m_part, m_curv, m_spir])

def run_unified_field():
    print("--- UNIFIED FIELD THEORY (THE ULTIMATE METRIC) ---")
    print("Modelling Nonce as: Mass, Wave, Particle, Curve, Spiral.")
    
    # 905561
    ver = 598728704
    prev_hex = "00000000000000000001c95188c655f79a281d351db7ffad034d39ba3c6be4ce"
    merkle_hex = "3c38914753b8b54b0fff74ca07e5c998e69523a4f3efa82a871075e46ee233ee"
    time_val = 1752527466
    bits = 386022054
    real_nonce = 3536931971
    
    header_tmpl = struct.pack("<I", ver) + binascii.unhexlify(prev_hex)[::-1] + \
                  binascii.unhexlify(merkle_hex)[::-1] + struct.pack("<I", time_val) + \
                  struct.pack("<I", bits) + b'\x00\x00\x00\x00'
                  
    physics = UnifiedPhysics()
    
    # Real
    v_real = physics.compute_super_metric(header_tmpl, real_nonce)
    print("\n[REAL NONCE PHYSICS]")
    print(f"Mass:     {v_real[0]:.4f}")
    print(f"Wave:     {v_real[1]:.4f}")
    print(f"Particle: {v_real[2]:.4f}")
    print(f"Curve:    {v_real[3]:.4f}")
    print(f"Spiral:   {v_real[4]:.4f}")
    print(f">> Magnitude: {np.linalg.norm(v_real):.4f}")
    
    # Random Field
    print("\n[QUANTUM VACUUM (RANDOM FIELD)]")
    cloud = []
    for _ in range(500):
        r = random.randint(0, 2**32-1)
        cloud.append(physics.compute_super_metric(header_tmpl, r))
    
    cloud = np.array(cloud)
    means = np.mean(cloud, axis=0)
    stds = np.std(cloud, axis=0)
    
    # Calculate Z-Scores
    z_scores = np.abs(v_real - means) / stds
    
    print("-" * 50)
    print(f"{'DIMENSION':<15} | {'REAL':<10} | {'VACUUM':<10} | {'SIGMA':<10}")
    print("-" * 50)
    dims = ["Mass", "Wave", "Particle", "Curve", "Spiral"]
    for i in range(5):
        print(f"{dims[i]:<15} | {v_real[i]:<10.4f} | {means[i]:<10.4f} | {z_scores[i]:<10.2f}")
    print("-" * 50)
    
    max_sigma = np.max(z_scores)
    print(f"\nMAXIMUM ANOMALY: {max_sigma:.2f} Sigma")
    
    if max_sigma > 3.0:
        print(">>> DISCOVERY: UNIFIED FIELD BREAKTHROUGH CONFIRMED!")
    else:
        print(">>> RESULT: Unified Vector is within standard quantum fluctuations.")

if __name__ == "__main__":
    run_unified_field()
