"""
CMFO Core Geometry
==================
Defines the structure of the T^7_phi manifold (fractal torus) and its metric.

This module is the authoritative source for:
- Scale parameters (PHI, LAMBDA)
- Distance functions (geodesic_distance)
- Isometries (translation, reflection)
"""

import numpy as np

# -----------------------------------------------------------------------------
# CONSTANTS & PARAMETERS
# -----------------------------------------------------------------------------

# Golden ratio: The fundamental scaling factor
PHI = (1 + np.sqrt(5)) / 2

# Fractal weights lambda_i = phi^(i-1) for i=0..6
# This defines the anisotropy of the metric g_phi.
LAMBDA = np.array([PHI**i for i in range(7)])

# -----------------------------------------------------------------------------
# METRIC & TOPOLOGY
# -----------------------------------------------------------------------------

def wrap_angle(theta: float) -> float:
    """
    Wrap angle to the canonical interval (-pi, pi].
    Ensures calculations on S^1 are uniquely defined.
    """
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

def geodesic_distance(theta: np.ndarray, eta: np.ndarray) -> float:
    """
    Calculates the geodesic distance d_phi(theta, eta) on the T^7_phi manifold.
    
    The metric is diagonal but anisotropic:
        d_phi^2 = Sum_i lambda_i * (theta_i - eta_i)^2
    where differences are taken modulo 2pi (on the torus).
    
    Args:
        theta (np.ndarray): Point 1 in R^7 (mod 2pi)
        eta (np.ndarray): Point 2 in R^7 (mod 2pi)
        
    Returns:
        float: The weighted Riemannian distance.
    """
    assert theta.shape == (7,) and eta.shape == (7,), "States must be 7-dimensional"
    
    # Calculate difference on the torus (minimal wrap)
    delta = np.array([wrap_angle(theta[i] - eta[i]) for i in range(7)])
    
    # Weighted Euclidean distance (Riemannian)
    return np.sqrt(np.sum(LAMBDA * delta**2))

# -----------------------------------------------------------------------------
# ISOMETRIES
# -----------------------------------------------------------------------------

def translation(theta: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Applies the translation operator T_a(theta) = theta + a.
    This is an isometry of the flat torus.
    """
    return (theta + shift) % (2 * np.pi)

def reflection(theta: np.ndarray, axis: int) -> np.ndarray:
    """
    Applies the reflection operator R_i(theta).
    Flips the sign of the *axis*-th coordinate.
    """
    result = theta.copy()
    result[axis] = -result[axis]
    return result % (2 * np.pi)
