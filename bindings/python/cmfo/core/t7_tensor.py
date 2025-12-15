# =====================================================================
# CMFO-COMPUTE - AVISO DE LICENCIA
# Uso académico y personal permitido bajo Apache 2.0.
# El uso comercial, corporativo o gubernamental requiere licencia CMFO.
# Contacto comercial:
#   Jonathan Montero Viquez – San José, Costa Rica
#   jmvlavacar@hotmail.com
# =====================================================================
import math
from .gamma_phi import gamma_step


class T7Tensor:
    def __init__(self, v):
        self.v = list(v) if not isinstance(v, list) else v

    def evolve(self, steps=1):
        v = self.v
        for _ in range(steps):
            v = gamma_step(v)
        return T7Tensor(v)

    def norm(self):
        return math.sqrt(sum(x * x for x in self.v))

    def __repr__(self):
        return f"T7Tensor({self.v})"
