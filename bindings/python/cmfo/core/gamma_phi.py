# =====================================================================
# CMFO-COMPUTE - AVISO DE LICENCIA
# Uso académico y personal permitido bajo Apache 2.0.
# El uso comercial, corporativo o gubernamental requiere licencia CMFO.
# Contacto comercial:
#   Jonathan Montero Viquez – San José, Costa Rica
#   jmvlavacar@hotmail.com
# =====================================================================
import math


from ..compiler.jit import FractalJIT
from ..compiler.ir import symbol, fractal_sin

def gamma_step(v):
    v = list(v) if not isinstance(v, list) else v
    
    # [GPU BRIDGE] Attempt JIT Execution
    if FractalJIT.is_available():
        try:
            # Build 7D IR Graph: out = sin(v)
            op = fractal_sin(symbol('v'))
            
            # Dummy h (hidden state) required by current Kernel Signature
            h = [0.0] * len(v)
            
            # Execute
            result_batch = FractalJIT.compile_and_run(op, v, h)
            if result_batch and len(result_batch) > 0:
                return result_batch[0]
        except Exception as e:
            # Silent fallback to CPU allows smooth degradation
            pass

    # CPU Fallback (The "Simulation")
    return [math.sin(x) for x in v]

