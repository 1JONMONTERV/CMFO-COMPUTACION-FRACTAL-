import cmfo


# Mapeo: 0 (Falso) -> -1, 1 (Verdadero) -> 1
def to_cmfo(bit):
    return 1.0 if bit else 0.0


def from_cmfo(val):
    # Handle complex numbers from fractal operations
    if hasattr(val, 'real'):
        return val.real > 0
    return val > 0


class TestBooleanAbsorption:
    """
    Verifica matemáticamente que los operadores continuos de CMFO
    absorben perfectamente la lógica booleana en los límites.
    """

    def test_completeness_and(self):
        """Verifica la tabla de verdad AND"""
        truth_table = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        for a, b, expected in truth_table:
            val_a = to_cmfo(a)
            val_b = to_cmfo(b)
            result = cmfo.f_and(val_a, val_b)
            assert from_cmfo(result) == expected, \
                f"AND Error: {a}&{b}!={expected}"

    def test_completeness_or(self):
        """Verifica la tabla de verdad OR"""
        truth_table = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        for a, b, expected in truth_table:
            val_a = to_cmfo(a)
            val_b = to_cmfo(b)
            result = cmfo.f_or(val_a, val_b)
            assert from_cmfo(result) == expected, \
                f"OR Error: {a}|{b}!={expected}"

    def test_completeness_xor(self):
        """Verifica la tabla de verdad XOR"""
        truth_table = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        for a, b, expected in truth_table:
            val_a = to_cmfo(a)
            val_b = to_cmfo(b)
            result = cmfo.f_xor(val_a, val_b)
            assert from_cmfo(result) == expected, \
                f"XOR Error: {a}^{b}!={expected}"

    def test_completeness_nand(self):
        """Verifica la tabla de verdad NAND (Funcionalmente Completo)"""
        truth_table = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        for a, b, expected in truth_table:
            val_a = to_cmfo(a)
            val_b = to_cmfo(b)
            
            # CMFO doesn't check 'phi_nand', we must implement NAND from AND+NOT
            # Or perhaps 'f_nand' exists?
            # Based on previous dir() listing, it does NOT exist.
            # We implemented NAND as NOT(AND(a,b))
            res_and = cmfo.f_and(val_a, val_b)
            result = cmfo.f_not(res_and)
            
            assert from_cmfo(result) == expected, \
                f"NAND Error: {a} NAND {b}!={expected}"

    def test_continuity_hypothesis(self):
        """
        Verifica que el operador funciona incluso con ruido.
        Esto demuestra la robustez del axioma de continuidad.
        """
        # 0.8 es "Casi Verdad" (True)
        assert cmfo.f_and(0.8, -0.9).real < 0  # True & False -> False
        assert cmfo.f_and(0.8, 0.2).real > 0   # True & Weak True -> True
