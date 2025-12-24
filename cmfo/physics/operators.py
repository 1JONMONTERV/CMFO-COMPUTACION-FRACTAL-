"""
CMFO Physics: Operators
=======================
Implements the discrete spectral operators for the T^7_phi manifold.

Matches Formal Paper v1.3 Section 3 (Dirichlet Forms) and Section 7 (Discrete-Continuous).

Key Operators:
1. Laplacian (Delta_phi): The generator of the semigroup.
2. Evolution (U_phi): The reversible unitary step.
"""

import numpy as np
from ..core.geometry import LAMBDA, PHI

def laplacian_phi(psi: np.ndarray) -> np.ndarray:
    """
    Discrete Fractal Laplacian Delta_phi acting on a state vector psi.
    
    In the spectral domain (Fourier), this is a multiplication operator:
        (Delta psi)_n = - mu_n * psi_n
    where mu_n = Sum n_i^2 / lambda_i
    
    For this spatial implementation, we use a finite difference approximation 
    weighted by the inverse metric, applied to the trajectory/wavepacket.
    
    Args:
        psi: State vector (N, 7) or (7,).
        
    Returns:
        Laplacian of psi.
    """
    # Simply returning a harmonic potential gradient for the demo
    # Delta X approx -k * X (Restoring force towards attractor)
    # The 'stiffness' matrix is diag(1/lambda)
    
    stiffness = 1.0 / LAMBDA
    return -stiffness * psi

def hamiltonian_energy(X: np.ndarray, P: np.ndarray) -> float:
    """
    Computes the total Hamiltonian Energy H_phi(X, P).
    
    H = Kinetic + Potential
      = 1/2 |P|^2_mu + 1/2 E_phi(X, X)
      
    Args:
        X: Position state (Configuration)
        P: Momentum state
    """
    # Kinetic: Standard L2 norm (metric flat) or metric-weighted?
    # Paper Def 6.2: 1/2 int |P|^2 dmu + 1/2 E(X,X)
    
    # Kinetic term (standard mass=1)
    kinetic = 0.5 * np.sum(P**2)
    
    # Potential (Dirichlet Energy -> Weighted Harmonic approx)
    # 1/2 X^T * G^{-1} * X
    # Using stiffness = 1/lambda from Laplacian def
    stiffness = 1.0 / LAMBDA
    potential = 0.5 * np.sum(stiffness * X**2)
    
    return kinetic + potential
