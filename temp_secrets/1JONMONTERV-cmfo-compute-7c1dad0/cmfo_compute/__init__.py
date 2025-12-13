# =====================================================================
# CMFO-COMPUTE - AVISO DE LICENCIA
# Uso académico y personal permitido bajo Apache 2.0.
# El uso comercial, corporativo o gubernamental requiere licencia CMFO.
# Contacto comercial:
#   Jonnathan Montero – San José, Costa Rica
#   jmvlavacar@hotmail.com
# =====================================================================
from .core.t7_tensor import T7Tensor
from .core.gamma_phi import gamma_step
from .logic.phi_logic import (
    phi_sign,
    phi_and,
    phi_or,
    phi_not,
    phi_xor,
    phi_nand,
)


def tensor(v):
    return T7Tensor(v)


__all__ = [
    "T7Tensor",
    "tensor",
    "gamma_step",
    "phi_sign",
    "phi_and",
    "phi_or",
    "phi_not",
    "phi_xor",
    "phi_nand",
]
