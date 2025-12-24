# =====================================================================
# CMFO-COMPUTE - AVISO DE LICENCIA
# Uso académico y personal permitido bajo Apache 2.0.
# El uso comercial, corporativo o gubernamental requiere licencia CMFO.
# Contacto comercial:
#   Jonathan Montero Viquez – San José, Costa Rica
#   jmvlavacar@hotmail.com
# =====================================================================


def phi_sign(x):
    # Convert to float if it's a list/iterable
    if isinstance(x, (list, tuple)):
        x = sum(x) / len(x) if x else 0.0
    x = float(x)
    return 1.0 if x >= 0 else -1.0


def phi_and(a, b):
    return min(phi_sign(a), phi_sign(b))


def phi_or(a, b):
    return phi_sign(a) if phi_sign(a) == 1 else phi_sign(b)


def phi_not(a):
    return -phi_sign(a)


def phi_xor(a, b):
    return 1.0 if phi_sign(a) != phi_sign(b) else -1.0



def _clamp_phi(val, lower=-1.0, upper=1.0):
    return max(lower, min(val, upper))


def fractal_xor(x, y):
    """
    XOR fractal: clamp(x + y - xy * |x+y|/2)
    """
    x, y = float(x), float(y)
    term = x + y - (x * y * abs(x + y) / 2.0)
    return _clamp_phi(term)


def fractal_and(x, y):
    """
    AND fractal: (x*y) * (1 - |x-y|/2)
    """
    x, y = float(x), float(y)
    return (x * y) * (1.0 - abs(x - y) / 2.0)


def fractal_or(x, y):
    """
    OR fractal: (x+y-xy) * (1 - |x+y-xy|/2)
    """
    x, y = float(x), float(y)
    base = x + y - (x * y)
    return base * (1.0 - abs(base) / 2.0)


def phi_nand(a, b):
    return -phi_and(a, b)
