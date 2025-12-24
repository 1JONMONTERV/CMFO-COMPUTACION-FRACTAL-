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


def phi_nand(a, b):
    return -phi_and(a, b)
