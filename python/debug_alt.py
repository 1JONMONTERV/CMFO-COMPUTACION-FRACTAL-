# Minimal debug test
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from cmfo.universal.octonion_algebra import Octonion, verify_alternativity

# Run test multiple times
for trial in range(5):
    result = verify_alternativity()
    print(f"Trial {trial}: left_err={result['left_alternative_error']:.6f}, right_err={result['right_alternative_error']:.6f}")
    print(f"         left_holds={result['left_holds']}, right_holds={result['right_holds']}")
