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

    def test_completeness_nor(self):
        """Verifica la tabla de verdad NOR (también funcionalmente completo)"""
        truth_table = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
        for a, b, expected in truth_table:
            val_a = to_cmfo(a)
            val_b = to_cmfo(b)
            
            # NOR = NOT(OR(a,b))
            res_or = cmfo.f_or(val_a, val_b)
            result = cmfo.f_not(res_or)
            
            assert from_cmfo(result) == expected, \
                f"NOR Error: {a} NOR {b}!={expected}"

    def test_completeness_xnor(self):
        """Verifica la tabla de verdad XNOR (equivalencia)"""
        truth_table = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        for a, b, expected in truth_table:
            val_a = to_cmfo(a)
            val_b = to_cmfo(b)
            
            # XNOR = NOT(XOR(a,b))
            res_xor = cmfo.f_xor(val_a, val_b)
            result = cmfo.f_not(res_xor)
            
            assert from_cmfo(result) == expected, \
                f"XNOR Error: {a} XNOR {b}!={expected}"

    def test_de_morgan_law_1(self):
        """Verifica primera ley de De Morgan: NOT(a AND b) = (NOT a) OR (NOT b)"""
        test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for a, b in test_cases:
            val_a = to_cmfo(a)
            val_b = to_cmfo(b)
            
            # Lado izquierdo: NOT(a AND b)
            lhs = cmfo.f_not(cmfo.f_and(val_a, val_b))
            
            # Lado derecho: (NOT a) OR (NOT b)
            rhs = cmfo.f_or(cmfo.f_not(val_a), cmfo.f_not(val_b))
            
            assert from_cmfo(lhs) == from_cmfo(rhs), \
                f"De Morgan 1 Error: a={a}, b={b}"

    def test_de_morgan_law_2(self):
        """Verifica segunda ley de De Morgan: NOT(a OR b) = (NOT a) AND (NOT b)"""
        test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for a, b in test_cases:
            val_a = to_cmfo(a)
            val_b = to_cmfo(b)
            
            # Lado izquierdo: NOT(a OR b)
            lhs = cmfo.f_not(cmfo.f_or(val_a, val_b))
            
            # Lado derecho: (NOT a) AND (NOT b)
            rhs = cmfo.f_and(cmfo.f_not(val_a), cmfo.f_not(val_b))
            
            assert from_cmfo(lhs) == from_cmfo(rhs), \
                f"De Morgan 2 Error: a={a}, b={b}"

    def test_absorption_law(self):
        """Verifica ley de absorción: a AND (a OR b) = a"""
        test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for a, b in test_cases:
            val_a = to_cmfo(a)
            val_b = to_cmfo(b)
            
            # a AND (a OR b) debería ser igual a a
            result = cmfo.f_and(val_a, cmfo.f_or(val_a, val_b))
            
            assert from_cmfo(result) == a, \
                f"Absorption Error: a={a}, b={b}"

    def test_continuity_hypothesis(self):
        """
        Verifica que el operador funciona incluso con ruido.
        Esto demuestra la robustez del axioma de continuidad.
        """
        # 0.8 es "Casi Verdad" (True)
        assert cmfo.f_and(0.8, -0.9).real < 0  # True & False -> False
        assert cmfo.f_and(0.8, 0.2).real > 0   # True & Weak True -> True
