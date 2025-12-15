import sys
sys.path.insert(0, 'bindings/python')

from cmfo.constants import H, HBAR, C
import math

# Electron data
M_e = 9.1093837015e-31  # kg
L_e = 2.4263102367e-12  # m

print("BEFORE FIX (using hbar):")
m_old = HBAR / (C * L_e)
print(f"  Mass = {m_old:.10e} kg")
print(f"  Error = {abs(m_old - M_e)/M_e * 100:.2f}%")
print(f"  Ratio = {m_old/M_e:.6f} (expected 1/2π = {1/(2*math.pi):.6f})")
print()

print("AFTER FIX (using h):")
m_new = H / (C * L_e)
print(f"  Mass = {m_new:.10e} kg")
print(f"  Error = {abs(m_new - M_e)/M_e * 100:.6f}%")
print(f"  Ratio = {m_new/M_e:.10f}")
print()

print(f"Improvement: {abs(m_old - M_e)/M_e * 100:.2f}% → {abs(m_new - M_e)/M_e * 100:.6f}%")
