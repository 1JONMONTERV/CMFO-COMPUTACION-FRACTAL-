from .solver import SistemaResolucionUniversal7D, ResolvedorUniversal7D
from .figures import (
    FiguraSedaptica7D,
    FiguraToroEscalonado7D,
    FiguraEsferaExotica7D,
    FiguraCuboOctonionico7D,
    FiguraCantor7D,
    FiguraPhi7D,
    FiguraManifoldG2,
    FigurasFundamentales7D
)
from .teleportation import TeleportacionOctonionicaImposible, demostrar_teleportacion_imposible
from .constants import tensor_metrico_fundamental_7d
from .operations import (
    producto_phi_cruz_7d, gradiente_phi_optimo_7d, 
    ecuacion_phi_cubica_no_asociativa, codificacion_phi_adica_ultra_infinita,
    generar_informacion_infinita
)
