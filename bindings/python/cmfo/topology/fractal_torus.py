"""
CMFO Topology - Fractal Torus (Toro Fractal)
============================================

Implements a 1024x1024 Binary Lattice on a Fractal Torus topology.
This structure interacts via a 7x7 recursive matrix kernel ("Matriz Fractal"),
projecting the 7D Phi-Manifold dynamics onto a 2D surface.

Key Features:
- 1024x1024 Binary State Grid (FractalRAM)
- 7x7 Convolution Kernel based on Golden Ratio (Phi) powers.
- Toroidal Boundary Conditions (Periodic).
- Geometric Measurements: Tensors, Attractors, Angles.
"""

import numpy as np
import math
from ..constants import PHI

class FractalTorus:
    def __init__(self, size=1024, kernel_size=7):
        self.size = size
        self.kernel_size = kernel_size
        
        # Initialize 1024x1024 binary grid (random start for chaos/attractor search)
        # In a strict "exact geometric" definitions, we might start with a seed from center.
        self.grid = np.random.randint(0, 2, (size, size), dtype=np.int8)
        
        # Build the 7x7 Fractal Kernel
        # The kernel represents the local geometry of the space.
        # We use powers of Phi to weight the interactions, creating a "Fractal Potential".
        self.kernel = self._build_phi_kernel(kernel_size)
        
        # History for Attractor detection
        self.history = []
        
    def _build_phi_kernel(self, k):
        """
        Constructs a k x k kernel where weights decay by Phi based on distance from center.
        This embodies the 'Metric' of the space defined in metric.py
        """
        center = k // 2
        kernel = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                # Chebyshev distance or Euclidean? 
                # "Fractal" often implies L-infinity or Manhattan in grids, 
                # but let's use Euclidean for "Exact Geometric" rotation invariance approximation
                dist = math.sqrt((i - center)**2 + (j - center)**2)
                if dist == 0:
                    kernel[i, j] = 1.0 # Self
                else:
                    # Weight decays as phi^(-dist)
                    kernel[i, j] = PHI ** (-dist)
                    
        # Normalize so that convolution measures density relative to Phi
        return kernel / np.sum(kernel)

    def step(self):
        """
        Evolve the cellular automata on the Torus.
        """
        from scipy.signal import convolve2d
        
        # 1. Convolve with Toroidal Boundary Conditions (boundary='wrap')
        # This physically models the "Toro" topology.
        potential = convolve2d(self.grid, self.kernel, mode='same', boundary='wrap')
        
        # 2. Apply Non-linear Fractal Activation
        # A simple binary threshold is standard CA. 
        # For "Fractal", we can use a mod or specific phi-resort.
        # Let's use a "Phi Resonance" rule: if potential is close to specific phi-harmonic nodes.
        # Simplifying for "Binary Maze" (Mara Binaria):
        # State flips if local energy exceeds threshold (Reaction-Diffusion like).
        
        # "Exact Definition": Let's try Game of Life-like but with Phi weights.
        # If density > 1/Phi^2 (approx 0.382) -> 1, else 0?
        # Or XOR logic?
        
        # User requested "Mara Binaria" (Binary Mesh).
        # We will use a rule that sustains fractal growth. 
        # Rule: Output 1 if potential is within a "Goldilocks" Phi band.
        # 1/Phi^2 < p < 1/Phi
        lower = 1.0 / (PHI**2)
        upper = 1.0 / PHI
        
        next_grid = np.logical_and(potential > lower, potential < upper).astype(np.int8)
        
        self.grid = next_grid
        return self.grid

    def measure_geometry(self):
        """
        Calculates exact geometric definitions: Tensors, Angles.
        """
        # 1. Metric Tensor Field (Approximation)
        # We define the local metric tensor trace as the local bit density
        # measured by the 7x7 kernel.
        from scipy.signal import convolve2d
        local_density = convolve2d(self.grid, self.kernel, mode='same', boundary='wrap')
        
        # 2. Gradients (Vectors)
        # Calculate gradients in X and Y directions (Periodic)
        grad_x = np.roll(self.grid, -1, axis=0) - np.roll(self.grid, 1, axis=0)
        grad_y = np.roll(self.grid, -1, axis=1) - np.roll(self.grid, 1, axis=1)
        
        # 3. Exact Angles
        # The angle of the gradient vector relative to the lattice.
        angles = np.arctan2(grad_y, grad_x)
        
        # 4. Attractors (Global State)
        current_hash = hash(self.grid.tobytes())
        is_attractor = current_hash in self.history
        self.history.append(current_hash)
        if len(self.history) > 100: self.history.pop(0) # Keep sliding window
        
        return {
            "tensor_trace_mean": np.mean(local_density),
            "tensor_trace_std": np.std(local_density),
            "mean_angle": np.mean(angles),
            "entropy": -np.mean(local_density * np.log(local_density + 1e-10)),
            "in_attractor": is_attractor
        }

    def run_simulation(self, steps=100):
        print(f"Initializing Fractal Torus ({self.size}x{self.size}) with 7x7 Kernel...")
        for i in range(steps):
            self.step()
            metrics = self.measure_geometry()
            if i % 10 == 0:
                print(f"Step {i}: Tensor={metrics['tensor_trace_mean']:.6f}, "
                      f"Angle={metrics['mean_angle']:.6f}, "
                      f"Attractor={metrics['in_attractor']}")
        
        return self.grid
