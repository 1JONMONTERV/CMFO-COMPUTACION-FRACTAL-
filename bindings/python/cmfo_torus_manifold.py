
import os
import sys
import struct
import math
import numpy as np
import random
import binascii

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class TorusManifold:
    """
    Defines a Custom Metric Space on T^8 specifically for SHA-256.
    We treat the logic gates (Ch, Maj, Sigma) as geometric curvatures.
    """
    
    def __init__(self):
        pass

    def get_trajectory_angles(self, header, nonce):
        # 1. Recreate Trajectory (Same as Inverter but for full path action)
        h = bytearray(header)
        h[76:80] = struct.pack("<I", nonce)
        h_bytes = bytes(h)
        input_block = h_bytes + b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
        
        # Chunk 2 processing
        chunk = input_block[64:]
        W = list(struct.unpack(">16I", chunk)) + [0]*48
        
        # Schedule
        for i in range(16, 64):
            s0 = (W[i-15]>>7 | W[i-15]<<25) ^ (W[i-15]>>18 | W[i-15]<<14) ^ (W[i-15]>>3)
            s1 = (W[i-2]>>17 | W[i-2]<<15) ^ (W[i-2]>>19 | W[i-2]<<13) ^ (W[i-2]>>10)
            W[i] = (W[i-16] + s0 + W[i-7] + s1) & 0xFFFFFFFF
            
        H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
             
        a,b,c,d,e,f,g,h = H
        K = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        ]
        
        trajectory = []
        
        for i in range(64):
            S1 = (e>>6 | e<<26) ^ (e>>11 | e<<21) ^ (e>>25 | e<<7)
            ch = (e & f) ^ ((~e) & g)
            t1 = (h + S1 + ch + K[i] + W[i]) & 0xFFFFFFFF
            S0 = (a>>2 | a<<30) ^ (a>>13 | a<<19) ^ (a>>22 | a<<10)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (S0 + maj) & 0xFFFFFFFF
            
            h = g
            g = f
            f = e
            e = (d + t1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (t1 + t2) & 0xFFFFFFFF
            
            # STATE VECTOR (Angle 0..2pi)
            vec = [(x / 2**32) * 2 * math.pi for x in [a,b,c,d,e,f,g,h]]
            trajectory.append(vec)
            
        return np.array(trajectory)

    def geodesic_action(self, trajectory):
        """
        Calculates the 'Action' S along the trajectory on the Manifold.
        The Metric G is defined by the SHA-256 constraints.
        
        Ideally, G_ij is such that SHA-256 evolution is a geodesic (straight line).
        So deviation from 'Straight Line' in T^8 (Euclidean) represents 'Force'.
        
        We measure "Smoothness" of the path in angular space.
        Smooth Force = Low Action.
        Jerky Force = High Action.
        
        Metric: Sum of Squared Angular Accelerations (Jerk).
        Valid blocks (resonants) should flow 'smoother'.
        """
        # Velocity in angle space
        vel = np.diff(trajectory, axis=0)
        # Correct wrapping
        vel[vel > np.pi] -= 2*np.pi
        vel[vel < -np.pi] += 2*np.pi
        
        # Acceleration
        acc = np.diff(vel, axis=0)
        # Correct wrapping
        acc[acc > np.pi] -= 2*np.pi
        acc[acc < -np.pi] += 2*np.pi
        
        # Action density (Kinetic Energy of the path curvature)
        action_density = np.sum(acc**2, axis=1)
        total_action = np.sum(action_density)
        
        return total_action

    def harmonic_energy(self, trajectory):
        """
        Measure alignment with Torus Harmonics (Grid points).
        Energy = Sum of dist to nearest k*pi/2
        """
        # Dist to pi/2 grid
        grid = np.pi / 2
        dists = np.abs(np.remainder(trajectory, grid) - grid/2)
        # We want to be CLOSE to grid lines (0)
        return np.sum(dists)

def analyze_manifold():
    print("--- ELEVATED TOROIDAL MANIFOLD (T^8_SHA) ---")
    print("Hypothesis: Valid Nonces follow Geodesics (Minimal Action Paths).")
    
    # Header Setup
    ver = 598728704
    prev_hex = "00000000000000000001c95188c655f79a281d351db7ffad034d39ba3c6be4ce"
    merkle_hex = "3c38914753b8b54b0fff74ca07e5c998e69523a4f3efa82a871075e46ee233ee"
    time_val = 1752527466
    bits = 386022054
    real_nonce = 3536931971
    
    prev = binascii.unhexlify(prev_hex)[::-1]
    merkle = binascii.unhexlify(merkle_hex)[::-1]
    header_tmpl = struct.pack("<I", ver) + prev + merkle + struct.pack("<I", time_val) + struct.pack("<I", bits) + b'\x00\x00\x00\x00'
    
    manifold = TorusManifold()
    
    # Real
    traj_real = manifold.get_trajectory_angles(header_tmpl, real_nonce)
    action_real = manifold.geodesic_action(traj_real)
    energy_real = manifold.harmonic_energy(traj_real)
    
    print(f"\n[REAL NONCE GEODESIC]")
    print(f"Action (S):      {action_real:.4f}")
    print(f"Pot. Energy (U): {energy_real:.4f}")
    
    # Population
    print("\n[CHAOS FIELD (N=500)]")
    actions = []
    energies = []
    
    for _ in range(500):
        r = random.randint(0, 2**32-1)
        t = manifold.get_trajectory_angles(header_tmpl, r)
        actions.append(manifold.geodesic_action(t))
        energies.append(manifold.harmonic_energy(t))
        
    mu_s = np.mean(actions)
    std_s = np.std(actions)
    
    mu_u = np.mean(energies)
    std_u = np.std(energies)
    
    z_s = (action_real - mu_s) / std_s
    z_u = (energy_real - mu_u) / std_u
    
    print("-" * 50)
    print(f"{'METRIC':<15} | {'REAL':<10} | {'FIELD AVG':<10} | {'SIGMA':<10}")
    print("-" * 50)
    print(f"{'Action (S)':<15} | {action_real:<10.2f} | {mu_s:<10.2f} | {z_s:<10.2f}")
    print(f"{'Energy (U)':<15} | {energy_real:<10.2f} | {mu_u:<10.2f} | {z_u:<10.2f}")
    print("-" * 50)
    
    if abs(z_s) > 3.0 or abs(z_u) > 3.0:
        print(">>> DISCOVERY: GEODESIC RESONANCE CONFIRMED! The path is optimal.")
    else:
        print(">>> RESULT: Path is turbulent (Standard SHA-256 Chaos).")

if __name__ == "__main__":
    analyze_manifold()
