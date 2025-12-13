import json
import numpy as np
import os
# Adapted import to match package name
from cmfo_compute.core.api import tensor7

def test_tensor7_equivalence():
    # Construct absolute path to data file to avoid CWD issues
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "golden_tensor7.json")
    
    with open(data_path) as f:
        golden = json.load(f)

    a = np.array(golden["input_a"])
    b = np.array(golden["input_b"])

    out = tensor7(a, b)
    ref = np.array(golden["output"])

    # Verify calculation matches golden reference
    assert np.allclose(out, ref, atol=1e-3), f"Output {out} does not match ref {ref}"
