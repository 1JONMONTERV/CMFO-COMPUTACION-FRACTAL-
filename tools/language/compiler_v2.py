import numpy as np
from typing import Dict, Any, List, Optional
import sys

# ==================== MOCKS for Dependencies ====================
class Tipo:
    N = "N"
    V = "V"
    DET = "Det"
    ADJ = "Adj"
    ADV = "Adv"
    CLITIC = "Clitic"
    CONJ = "C"
    PREP = "P"

class NodoArbol:
    def __init__(self, valor: str, tipo: str, hijos: List['NodoArbol'] = None, operador: str = None):
        self.valor = valor
        self.tipo = tipo
        self.hijos = hijos if hijos else []
        self.operador = operador

class ParserAlgebraicoCMFO:
    """
    Simulated Parser for the V2 Demo.
    Converts sentences into dependency trees based on hardcoded grammar rules for the demo.
    """
    def parsear(self, oracion: str) -> Optional[NodoArbol]:
        tokens = oracion.lower().split()
        
        # Rule 1: "Juan estudia física" (S V O)
        if tokens == ["juan", "estudia", "física"]:
            # Tree: (SUJ Juan) (V estudia (OBJ física))
            # Actually V2 Semantic: SUJ(Juan) @ V(estudia) @ OBJ(física)
            # Simplified for demo:
            subj = NodoArbol("juan", Tipo.N)
            obj = NodoArbol("física", Tipo.N) # Assuming 'física' is N
            verb = NodoArbol("estudia", Tipo.V, [obj], operador="APP_OBJ") # verb applies to obj
            
            # Root: Subj apply to Predicate
            return NodoArbol("juan_estudia_fisica", Tipo.V, [subj, verb], operador="APP_SUJ")

        # Rule 2: "Juan lo ve" (S Clitic V)
        if tokens == ["juan", "lo", "ve"]:
            subj = NodoArbol("juan", Tipo.N)
            clitic = NodoArbol("lo", Tipo.CLITIC)
            verb = NodoArbol("ve", Tipo.V)
            
            # Clitic modifies Verb: CLITIC @ V
            verb_phrase = NodoArbol("lo_ve", Tipo.V, [clitic, verb], operador="CLITIC")
            
            # SUJ @ VP
            return NodoArbol("juan_lo_ve", Tipo.V, [subj, verb_phrase], operador="APP_SUJ")

        # Rule 3: "el niño pequeño lee" (Det N Adj V)
        if tokens == ["el", "niño", "pequeño", "lee"]:
            det = NodoArbol("el", Tipo.DET)
            noun = NodoArbol("niño", Tipo.N) # 'niño' not in base, will fallback
            adj = NodoArbol("pequeño", Tipo.ADJ) # 'pequeño' not in base
            verb = NodoArbol("lee", Tipo.V) # 'lee' not in base
            
            # NP construction: APP_DET(Det, MOD_ADJ(Adj, N))
            # Note: order of ops depends on linguistic theory.
            # Let's say: Det @ (Adj @ N)
            mod_n = NodoArbol("pequeño_niño", Tipo.N, [adj, noun], operador="MOD_ADJ")
            np_node = NodoArbol("el_niño", Tipo.N, [det, mod_n], operador="APP_DET")
            
            return NodoArbol("frase", Tipo.V, [np_node, verb], operador="APP_SUJ")

        return None

# ==================== USER'S V2 CODE ====================

class CompiladorCMFO_V2:
    """
    Compilador matricial: cada palabra = matriz 7x7
    Composición = multiplicación matricial (no suma)
    """
    
    def __init__(self):
        # Diccionario semántico: forma -> matriz de significado
        self.semantica: Dict[str, np.matrix] = self._inicializar_semantica()
        
        # Operadores de construcción sintáctica
        self.construccion = self._inicializar_operadores()
        
    def _inicializar_semantica(self) -> Dict[str, np.matrix]:
        """
        Cada palabra es un operador que transforma el espacio T7
        Las matrices son unitarias (conservan norma)
        """
        
        # Base canónica: cada tipo tiene una matriz identidad escalada
        base = {
            "N": np.matrix(np.diag([1.0, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]), dtype=complex),
            "V": np.matrix(np.diag([0.2, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]), dtype=complex),
            "Det": np.matrix(np.diag([0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1]), dtype=complex),
            "Adj": np.matrix(np.diag([0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1]), dtype=complex),
            "Adv": np.matrix(np.diag([0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1]), dtype=complex),
            "C": np.matrix(np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1]), dtype=complex),
            "P": np.matrix(np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]), dtype=complex),
        }
        
        # Nombres propios: matriz específica (tipo + referencia)
        sem = {}
        
        # Juan: NP masculino singular
        sem["juan"] = base["N"] * 1.2
        sem["juan"][0, 0] = 1.5  # Refuerza dimensión nominal
        
        # María: NP femenino
        sem["maria"] = base["N"] * 1.2
        sem["maria"][0, 0] = 1.5
        sem["maria"][0, 0] += 0.1j  # Marca femenino como fase
        
        # Verbos transitivos
        sem["ve"] = base["V"] @ np.matrix([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ])
        
        # Estudiar: requiere objeto acusativo
        sem["estudia"] = base["V"] @ np.matrix([
            [0.8,0,0,0,0,0,0],
            [0,1.2,0,0,0,0,0],  # Verbo más fuerte
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ])
        
        # Determinantes
        sem["el"] = base["Det"]
        sem["la"] = base["Det"] * (1 + 0.1j)  # Fase femenina
        sem["un"] = base["Det"] * 0.8
        
        # Clíticos pronominales: permutadores
        sem["lo"] = np.matrix([
            [1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0],  # Verbo -> objeto
            [0,1,0,0,0,0,0],  # Objeto -> verbo
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ])
        
        return sem
    
    def _inicializar_operadores(self) -> Dict[str, np.matrix]:
        """Operadores composicionales (funciones de orden superior)"""
        
        # APP_DET: Det + N -> NP
        # Multiplica las matrices y reordena
        APP_DET = np.matrix([
            [1.2,0,0,0,0,0,0],
            [0,0.9,0,0,0,0,0],
            [0,0,0,0,0,0,0],  # Det se consume
            [0,0,0,1.1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ])
        
        # CLITIC: Fusiona clítico con verbo
        CLITIC = np.matrix([
            [1,0,0,0,0,0,0],
            [0,np.exp(1j*np.pi/4),0,0,0,0,0],  # 45deg en V
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ])
        
        return {
            "APP_DET": APP_DET,
            "APP_SUJ": np.identity(7) * 1.0, # Subject Application
            "APP_OBJ": np.identity(7) * 1.0, # Object Application
            "CLITIC": CLITIC,
            "MOD_ADJ": np.identity(7) * 1.1,
            "MOD_ADV": np.identity(7) * 1.05,
            "SUJ": np.identity(7) * 1.0,
        }
    
    def compilar(self, arbol: NodoArbol) -> np.matrix:
        """
        Recorrido postorden: compone matrices mediante multiplicación
        """
        
        def recorrer(nodo: NodoArbol) -> np.matrix:
            # Hoja: buscar matriz semántica
            if not nodo.hijos:
                forma = nodo.valor.lower()
                if forma in self.semantica:
                    return self.semantica[forma]
                
                # Default: matriz identidad con peso tipo
                peso = 0.5 if nodo.tipo == Tipo.N else 0.8
                return np.matrix(np.identity(7) * peso)
            
            # Nodo interno: componer hijos
            hijos_compilados = [recorrer(hijo) for hijo in nodo.hijos]
            
            # Operador conocido: aplicar transformación
            if nodo.operador in self.construccion:
                op = self.construccion[nodo.operador]
                
                if len(hijos_compilados) == 2:
                    # Composición funcional: f @ g (Simplified logic for demo)
                    # For SUJ + VP -> VP @ SUJ or similar?
                    # Matrix mul is non-commutative. 
                    # Let's blindly follow: op @ h[0] @ h[1]
                    return op @ hijos_compilados[0] @ hijos_compilados[1]
                elif len(hijos_compilados) == 1:
                    return op @ hijos_compilados[0]
            
            # Operador desconocido: multiplicación directa
            resultado = np.identity(7)
            for h in hijos_compilados:
                resultado = resultado @ h
            
            return resultado
        
        return recorrer(arbol)
    
    def ejecutar_pipeline(self, oracion: str, parser: ParserAlgebraicoCMFO) -> Dict[str, Any]:
        """
        Pipeline completo con métricas de compilación
        """
        arbol = parser.parsear(oracion)
        
        if not arbol:
            return {"error": "No parseable"}
        
        matriz = self.compilar(arbol)
        
        # Métricas
        traza = complex(np.trace(matriz))
        determinante = complex(np.linalg.det(matriz))
        eigenvals = np.linalg.eigvals(matriz)
        
        return {
            "oracion": oracion,
            "matriz": matriz,
            "forma": matriz.shape,
            "traza": traza,
            "det": determinante,
            "rank": np.linalg.matrix_rank(matriz),
            "es_unitario": np.allclose(matriz @ matriz.H, np.identity(7)),
            "eigenvalues": eigenvals,
        }

# ==================== DEMO V2 ====================
def demo_v2():
    """Comparación V1 (vector) vs V2 (matriz)"""
    
    parser = ParserAlgebraicoCMFO()
    compilador_matricial = CompiladorCMFO_V2()
    
    oraciones = [
        "Juan estudia física",
        "Juan lo ve",
        "el niño pequeño lee",
    ]
    
    print("=" * 70)
    print("COMPILADOR MATRICIAL V2 - SEMÁNTICA OPERACIONAL")
    print("=" * 70)
    
    for oracion in oraciones:
        print(f"\nOración: '{oracion}'")
        print("-" * 70)
        
        resultado = compilador_matricial.ejecutar_pipeline(oracion, parser)
        
        if "error" in resultado:
            print("[X] No parseable")
            continue
        
        print(f"[OK] Matriz 7x7 compilada")
        print(f"  Traza: {resultado['traza']:.4f}")
        print(f"  Determinante: {resultado['det']:.4f}")
        print(f"  Rango: {resultado['rank']}")
        print(f"  Unitario: {resultado['es_unitario']}")
        
        # Mostrar matriz
        print("\n  Matriz resultante:")
        mat_str = np.array2string(resultado['matriz'], precision=3, suppress_small=True)
        for linea in mat_str.split('\n'):
            print(f"  {linea}")
        
        # Eigenvalores
        print(f"\n  Eigenvalores: {np.round(resultado['eigenvalues'], 3)}")

if __name__ == "__main__":
    demo_v2()
