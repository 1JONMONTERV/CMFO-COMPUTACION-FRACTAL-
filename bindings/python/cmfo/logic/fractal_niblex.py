
import math
import sys
import os

# Ensure we can import locally if running standalone
try:
    from .phi_logic import fractal_and, fractal_or, fractal_xor, phi_sign
except ImportError:
    # Fallback for dev/test without package install
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from logic.phi_logic import fractal_and, fractal_or, fractal_xor, phi_sign

PHI = (1 + math.sqrt(5)) / 2

class FractalNiblex11:
    """
    Niblex de 11 Trits (Balanced Ternary Fractal).
    Capacity: 3^11 = 177,147 states.
    Mapping: 11 bits -> 11 trits (usually 0->-1, 1->+1).
    """

    def __init__(self, trits=None):
        if trits is None:
            self.trits = [0] * 11
        else:
            # Validate trits are roughly in [-1, 1]
            if len(trits) != 11:
                raise ValueError("FractalNiblex11 requires exactly 11 trits.")
            self.trits = [float(t) for t in trits]

    @classmethod
    def from_int(cls, value, map_bools=True):
        """
        Create from integer (0..2047).
        If map_bools is True: 0 bit -> -1, 1 bit -> +1.
        If map_bools is False: 0 bit -> 0, 1 bit -> +1.
        """
        trits = []
        for i in range(11):
            bit = (value >> i) & 1
            if map_bools:
                trits.append(1.0 if bit else -1.0)
            else:
                trits.append(float(bit))
        return cls(trits)

    def phi_metric(self):
        """
        V(s) = Sum(s_i * phi^(-(i+1)))
        Returns the unique fractal identifier (float).
        """
        val = 0.0
        # Use 1-based indexing for powers to ensure decay < 1 start
        # phi^-1 = 0.618...
        current_phi = 1.0 / PHI
        for t in self.trits:
            val += t * current_phi
            current_phi /= PHI
        return val

    def apply(self, other, op_func):
        """
        Apply a fractal operator element-wise.
        Returns new FractalNiblex11.
        """
        new_trits = [op_func(a, b) for a, b in zip(self.trits, other.trits)]
        return FractalNiblex11(new_trits)

    def __and__(self, other):
        return self.apply(other, fractal_and)
    
    def __or__(self, other):
        return self.apply(other, fractal_or)

    def __xor__(self, other):
        return self.apply(other, fractal_xor)

    def collapse(self, threshold=0.0):
        """
        Collapses continuous values to {-1, 0, 1}.
        """
        collapsed = []
        for t in self.trits:
            if t > threshold:
                collapsed.append(1.0)
            elif t < -threshold:
                collapsed.append(-1.0)
            else:
                collapsed.append(0.0)
        return FractalNiblex11(collapsed)

    def __repr__(self):
        # Format as list of shortened floats or symbols
        symbols = []
        for t in self.trits:
            if abs(t - 1) < 0.1: symbols.append("+")
            elif abs(t + 1) < 0.1: symbols.append("-")
            elif abs(t) < 0.1: symbols.append("0")
            else: symbols.append(f"{t:.1f}")
        return f"Niblex11[{''.join(symbols)}] V={self.phi_metric():.4f}"

if __name__ == "__main__":
    # Quick demo
    import random
    
    print("--- Fractal Niblex 11 Demo ---")
    
    n1 = FractalNiblex11.from_int(2047) # All 1s
    n2 = FractalNiblex11.from_int(0)    # All 0s (mapped to -1s)
    
    print(f"N1 (All True): {n1}")
    print(f"N2 (All False): {n2}")
    
    nx = n1 ^ n2 # XOR
    print(f"N1 XOR N2: {nx} (Expect Zeros/Transitional?)")
    
    # Check Metric uniqueness
    print(f"Metric N1: {n1.phi_metric()}")
    print(f"Metric N2: {n2.phi_metric()}")
