# =====================================================================
# CMFO-COMPUTE - AVISO DE LICENCIA
# Uso académico y personal permitido bajo Apache 2.0.
# El uso comercial, corporativo o gubernamental requiere licencia CMFO.
# Contacto comercial:
#   Jonathan Montero Viquez – San José, Costa Rica
#   jmvlavacar@hotmail.com
# =====================================================================
import math


def gamma_step(v):
    v = list(v) if not isinstance(v, list) else v
    return [math.sin(x) for x in v]

