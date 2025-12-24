"""
CMFO Physics - Mass Relations
==============================

Geometric mass calculations based on Compton wavelength.
"""

import math
from ..constants import HBAR, C


def geometric_mass(L: float) -> float:
    """
    Geometric mass from characteristic length.
    
    Definition:
        m = h / (c · L)  (where h = 2π·ħ)
    
    This is the Compton relation: mass is inversely proportional to
    characteristic length scale.
    
    Parameters
    ----------
    L : float
        Characteristic length [m] (must be positive)
    
    Returns
    -------
    float
        Mass [kg]
    
    Raises
    ------
    ValueError
        If L is non-positive
    """
    if L <= 0:
        raise ValueError("Length must be positive.")
    
    # Corrected formula: use h instead of h_bar for standard Compton wavelength L
    return (HBAR * 2 * math.pi) / (C * L)


def compton_wavelength(m: float) -> float:
    """
    Compton wavelength from mass (inverse of geometric_mass).
    
    Definition:
        λ = ħ / (m · c)
    
    Parameters
    ----------
    m : float
        Mass [kg] (must be positive)
    
    Returns
    -------
    float
        Compton wavelength [m]
    
    Raises
    ------
    ValueError
        If m is non-positive
    """
    if m <= 0:
        raise ValueError("Mass must be positive.")
    
    return (HBAR * 2 * math.pi) / (m * C)


__all__ = ['geometric_mass', 'compton_wavelength']
