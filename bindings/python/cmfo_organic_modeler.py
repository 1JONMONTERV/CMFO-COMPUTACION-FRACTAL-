
import os
import sys
import struct
import math
import numpy as np
import random
import scipy.ndimage
import binascii

class OrganicModeler:
    """
    Biological Metaphors for Cryptocurrency Mining.
    """
    
    def __init__(self):
        pass

    def get_bit_grid(self, header, nonce):
        # Construct full block 640 bits... or just the Hash (256 bits)
        # Using the result Hash (H2) as the "Organism"
        # Mocking hash generation for speed (using simple sha256 lib to verify structure)
        # Actually, let's just use the Input Block (Header + Nonce) as the DNA.
        h = bytearray(header)
        h[76:80] = struct.pack("<I", nonce)
        # 80 bytes = 640 bits.
        # Reshape to 20x32 grid
        bits = []
        for byte in h:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
        
        return np.array(bits).reshape(20, 32)

    # --- MODEL 1: NEURON (Spike Response) ---
    def model_neuron(self, grid):
        # Treat grid as inputs to a random recurrent network.
        # Does it trigger a resonance/spike?
        # Simple Reservoir Computing proxy:
        # Random weights
        np.random.seed(42) # Consistent "Brain"
        weights = np.random.randn(20, 32)
        activation = np.sum(grid * weights)
        # Sigmoid
        spike_prob = 1.0 / (1.0 + np.exp(-activation))
        return spike_prob

    # --- MODEL 2: DISEASE (Viral Percolation) ---
    def model_disease(self, grid):
        # Treat 1s as infected cells.
        # Simulate simple SIR spread (Cellular Automata)
        # 1 step of spread: Infected cell infects neighbor with P=0.5
        # Measure "Epidemic Duration" or "Final Size"
        infected = grid.copy()
        for _ in range(5): # 5 generations
            # Dilate (spread to neighbors)
            growth = scipy.ndimage.binary_dilation(infected).astype(int)
            # Some recover (turn 0)? No, SIR usually R is removed.
            # Let's measure pure growth rate (Viral Load)
            infected = growth
        
        return np.sum(infected) / (20 * 32) # Saturation

    # --- MODEL 3: WHITE BLOOD CELL (Immune Affinity) ---
    def model_immune(self, grid):
        # Negative Selection: Distance from "Self" (Avg Header Structure)
        # We assume "Self" is the template without nonce.
        # Let's measure "Foreignness" (Contrast).
        # Simply: How distinct is the texture?
        # Using GLCM Contrast?
        # Or simpler: Local Binary Pattern (LBP) histogram entropy?
        # Let's use simple LBP proxy:
        # Count transitions 0->1 in both axes.
        transitions = np.sum(np.abs(np.diff(grid, axis=0))) + np.sum(np.abs(np.diff(grid, axis=1)))
        return transitions / (20 * 32)

    # --- MODEL 4: SKIN (Texture Analysis) ---
    def model_skin(self, grid):
        # Fractal Dimension of the surface (Box Counting on the grid)
        # Simplify: Minkowski-Bouligand
        # Scale 1: Count filled cells
        n1 = np.sum(grid)
        # Scale 2: Downsample 2x2 (max pool)
        h, w = grid.shape
        grid2 = grid.reshape(h//2, 2, w//2, 2).max(axis=(1,3))
        n2 = np.sum(grid2)
        
        if n2 == 0: return 0
        # Slope
        # log(N) vs log(1/s)
        # s1=1, N1. s2=2, N2.
        slope = (math.log(n2) - math.log(n1)) / (math.log(2) - math.log(1))
        return -slope # Dimension roughly

    def analyze(self):
        print("--- ORGANIC BIOLOGICAL MODELING ---")
        print("Models: Neuron (Spike), Disease (Viral), Immune (Foreignness), Skin (Texture)")
        
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
                      
        grid_real = self.get_bit_grid(header_tmpl, real_nonce)
        
        m_neuron = self.model_neuron(grid_real)
        m_disease = self.model_disease(grid_real)
        m_immune = self.model_immune(grid_real)
        m_skin = self.model_skin(grid_real)
        
        v_real = np.array([m_neuron, m_disease, m_immune, m_skin])
        
        print("\n[REAL ORGANISM]")
        print(f"Neuron Spike Prob: {m_neuron:.4f}")
        print(f"Viral Saturation:  {m_disease:.4f}")
        print(f"Immune Activity:   {m_immune:.4f}")
        print(f"Skin Fractal Dim:  {m_skin:.4f}")
        
        # Population
        print("\n[POPULATION CONTROL (N=500)]")
        pop = []
        for _ in range(500):
            r = random.randint(0, 2**32-1)
            g = self.get_bit_grid(header_tmpl, r)
            pop.append([
                self.model_neuron(g),
                self.model_disease(g),
                self.model_immune(g),
                self.model_skin(g)
            ])
            
        pop = np.array(pop)
        means = np.mean(pop, axis=0)
        stds = np.std(pop, axis=0)
        
        z_scores = np.abs(v_real - means) / stds
        
        print("-" * 60)
        names = ["Neuron", "Disease", "Immune", "Skin"]
        max_sig = 0
        best_bio = ""
        
        print(f"{'TYPE':<15} | {'REAL':<10} | {'POP AVG':<10} | {'SIGMA':<10}")
        print("-" * 60)
        for i in range(4):
            print(f"{names[i]:<15} | {v_real[i]:<10.4f} | {means[i]:<10.4f} | {z_scores[i]:<10.2f}")
            if z_scores[i] > max_sig:
                max_sig = z_scores[i]
                best_bio = names[i]
        print("-" * 60)
        
        print(f"\nDOMINANT BIOLOGICAL TRAIT: {best_bio} ({max_sig:.2f} Sigma)")
        
        if max_sig > 3.0:
            print(">>> LIFE DETECTED: Organism is distinct from background noise!")
        else:
            print(">>> RESULT: Organism blends with the population (Camouflage).")

if __name__ == "__main__":
    modeler = OrganicModeler()
    modeler.analyze()
