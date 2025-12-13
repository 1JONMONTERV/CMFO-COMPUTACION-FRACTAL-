import json
import numpy as np
import cmfo

def generate_golden_data():
    data = {
        "gamma_step": [],
        "phi_logic": []
    }
    
    # Test cases for Gamma Step (sin(x))
    inputs = [0.0, 0.5, 1.0, 3.14159, -1.0]
    for x in inputs:
        # Create tensor, evolve 1 step, take norm
        t = cmfo.tensor([x])
        evolved = t.evolve(1)
        expected = float(evolved.v[0])
        data["gamma_step"].append({
            "input": x,
            "expected_output": expected
        })

    # Test cases for Logic
    logic_inputs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    for a, b in logic_inputs:
        data["phi_logic"].append({
            "a": a,
            "b": b,
            "and": cmfo.phi_and(a, b),
            "or": cmfo.phi_or(a, b),
            "xor": cmfo.phi_xor(a, b),
            "nand": cmfo.phi_nand(a, b)
        })

    with open("golden_vectors.json", "w") as f:
        json.dump(data, f, indent=4)
        print("Generated golden_vectors.json")

if __name__ == "__main__":
    generate_golden_data()
