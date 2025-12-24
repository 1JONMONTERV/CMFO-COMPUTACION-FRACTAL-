
import random
import math
import sys
import statistics
sys.path.insert(0, '../bindings/python')
from cmfo.core.structural import FractalVector7
from cmfo.compiler.jit import FractalJIT

class AutopoieticUniverse:
    """
    Simulation Engine for a single evolutionary timeline.
    """
    def __init__(self, seed_phi=None, seed_alpha=None, noise_level=0.0):
        self.phi = seed_phi if seed_phi else (1 + math.sqrt(5))/2
        self.alpha = seed_alpha if seed_alpha else 0.0072973525 # Default if None
        self.noise = noise_level
        self.history = [] # Stores (alpha, energy) per generation
        self.generation = 0
        self.current_field = [0.1] * 7 # Initial Vacuum
        self.dummy_h = [0.0] * 7

    def step(self):
        """
        Executes one evolutionary step (Generation).
        Returns: (new_alpha, energy_density)
        """
        # 1. Define Law based on current parameters (Autopoiesis)
        # Equation: Field_new = Field_old * Alpha + Phi (Simplified Model)
        v = FractalVector7.symbolic('v')
        h = FractalVector7.symbolic('h')
        
        # Add noise to interaction if enabled
        effective_alpha = self.alpha + random.gauss(0, self.noise) if self.noise > 0 else self.alpha
        
        # Mutation logic: Laws might flip geometry based on parity to simulate complex dynamics
        if self.generation % 2 == 0:
            eq = v * effective_alpha + self.phi + (h * 0.0)
        else:
            eq = v * effective_alpha - (1.0/self.phi) + (h * 0.0)
            
        # 2. Compile & Run
        # Input is previous field state
        # Flatten input if it's a list of lists (from previous JIT output)
        flat_input = self.current_field
        if isinstance(flat_input[0], list):
             flat_input = [x for row in flat_input for x in row]
             
        # Run JIT
        try:
            res = FractalJIT.compile_and_run(eq._node, flat_input, self.dummy_h)
        except Exception as e:
            # Handle potential overflow/divergence
            return None 

        # 3. Measure (Feedback)
        flat_res = [x for row in res for x in row]
        energy = sum(x**2 for x in flat_res) / len(flat_res)
        
        # Limit energy to avoid infinity in simulation
        if energy > 1e6: energy = 1e6
        
        # 4. Evolve Parameters (The Feedback Loop)
        # Hypothesis: Coupling (Alpha) adapts to minimize total energy (Action principle)
        # Alpha_new = 1 / (Energy + Constant)
        new_alpha = 1.0 / (energy + 137.0) 
        
        # Update State
        self.alpha = new_alpha
        self.current_field = flat_res
        self.generation += 1
        
        self.history.append({'gen': self.generation, 'alpha': self.alpha, 'energy': energy})
        return self.alpha, energy

    def run(self, steps=20):
        for _ in range(steps):
            res = self.step()
            if res is None: break
        return self.history
