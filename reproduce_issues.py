
import sys
import os

# Ensure we can import the local package
sys.path.insert(0, os.path.abspath("bindings/python"))

print("=== CMFO Manual Verification ===\n")

try:
    print("1. Testing Logic API names...")
    import cmfo.logic as logic
    print(f"  Available attributes in cmfo.logic: {[x for x in dir(logic) if not x.startswith('__')]}")
    
    # Check for the function names that caused errors
    if hasattr(logic, 'phi_or'):
        print("  [OK] 'phi_or' exists.")
    elif hasattr(logic, 'f_or'):
        print("  [INFO] 'phi_or' NOT found. Found 'f_or' instead. The test suite is outdated.")
    else:
        print("  [FAIL] Neither 'phi_or' nor 'f_or' found.")

except ImportError as e:
    print(f"  [CRITICAL] Could not import cmfo.logic: {e}")
except Exception as e:
    print(f"  [ERROR] {e}")

print("\n2. Testing Physics Accuracy (Electron Mass)...")
try:
    from cmfo.physics import geometric_mass, compton_wavelength
    
    # Constants from the test
    L_e_input = 2.4263102367e-12
    m_e_expected = 9.1093837015e-31
    
    m_e_calculated = geometric_mass(L_e_input)
    
    print(f"  Input L_e: {L_e_input} m")
    print(f"  Expected Mass: {m_e_expected} kg")
    print(f"  Calculated Mass: {m_e_calculated} kg")
    
    error = abs(m_e_calculated - m_e_expected)
    relative_error = error / m_e_expected
    
    print(f"  Absolute Error: {error}")
    print(f"  Relative Error: {relative_error:.4f} ({relative_error*100:.2f}%)")
    
    if relative_error > 0.01:
        print("  [FAIL] Physics calculation is wildly inaccurate.")
    else:
        print("  [PASS] Physics calculation is accurate.")

except ImportError:
     print("  [CRITICAL] Could not import cmfo.physics")
except Exception as e:
    print(f"  [ERROR] {e}")

