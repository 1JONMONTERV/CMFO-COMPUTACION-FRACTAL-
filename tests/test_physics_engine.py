"""
TEST: CMFO Physics Engine
=========================
Verifies the fundamental dynamic laws of the system.
Matches "Certificación Computacional" section of v1.3 paper.

Claims Tested:
1. Energy Conservation: H(X_n) approx constant (Order dt^2).
2. Reversibility: Time reversal integration recovers initial state.
"""

import pytest
import numpy as np
from cmfo.physics import SymplecticIntegrator, hamiltonian_energy

class TestPhysicsEngine:
    
    def setup_method(self):
        self.integrator = SymplecticIntegrator(dt=0.01)

    def test_energy_conservation(self):
        """
        CLAIM: "Sistemas Conservativos"
        Energy should oscillate but not drift (symplectic property).
        """
        # Initial State (Random Phase Point)
        X0 = np.random.randn(7)
        P0 = np.random.randn(7)
        
        # Initial Energy
        E0 = hamiltonian_energy(X0, P0)
        
        # Evolve 1000 steps
        traj = self.integrator.evolve_trajectory(X0, P0, steps=1000)
        
        # Check final energy
        X_final, P_final = traj[-1]
        E_final = hamiltonian_energy(X_final, P_final)
        
        # Energy variation should be small (bounded by O(dt^2))
        variation = abs(E_final - E0) / abs(E0)
        
        assert variation < 1e-3, \
             f"Energy drift too high: {variation:.5f} (Must be < 0.1%)"

    def test_time_reversibility(self):
        """
        CLAIM: "Reversibilidad Exacta"
        Injecting -P at t=T should recover X0 at t=2T.
        """
        X0 = np.random.randn(7)
        P0 = np.random.randn(7)
        
        # Forward 100 steps
        traj = self.integrator.evolve_trajectory(X0, P0, steps=100)
        X_T, P_T = traj[-1]
        
        # Reverse momentum
        P_reverse = -P_T
        
        # Backward 100 steps
        traj_back = self.integrator.evolve_trajectory(X_T, P_reverse, steps=100)
        X_final, P_final = traj_back[-1]
        
        # Should match X0
        # Note: Symplectic methods are reversible.
        dist = np.linalg.norm(X_final - X0)
        
        assert dist < 1e-10, \
            f"Reversibility failed. Distance from origin: {dist}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
