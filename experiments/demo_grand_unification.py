
import sys
import os
import cmfo # Asumimos que está instalado o en path

# Asegurar path para imports locales si cmfo no está instalado globalmente
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core", "python"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))

try:
    from cmfo import fractal_add, fractal_product, fractal_root, phi_decision, geometric_state_collapse
except ImportError:
    # Si falla, definimos stubs para que la demo corra conceptualmente
    print("WARNING: No se pudo importar cmfo core. Usando stubs.")
    def fractal_add(a, b): return a + b
    def fractal_product(a, b): return a * b
    def fractal_root(a): return a ** 0.5
    def phi_decision(a): return 0
    def geometric_state_collapse(a): return 0

from cmfo.crypto.sha256d_reversible import sha256d_forward_reversible

class SpanishAlgebraCorrected:
    """
    Versión corregida de Spanish Algebra usando los operadores de cmfo 1.1.0
    """
    def __init__(self):
        self.operators = {
            'suma': fractal_add,
            'más': fractal_add,
            'resta': lambda a, b: fractal_add(a, -b), # Simplificación
            'multiplica': fractal_product,
            'por': fractal_product,
            'divide': lambda a, b: fractal_product(a, 1/b if b!=0 else 0), # Simplificación
        }
        self.numbers = {
            'cero': 0, 'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4,
            'cinco': 5, 'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9,
            'diez': 10, 'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14,
            'quince': 15, 'dieciséis': 16, 'veinte': 20, 'treinta': 30,
            'cuarenta': 40, 'cincuenta': 50, 'cien': 100, 'mil': 1000,
        }
        
    def parse_number(self, text):
        text = text.strip().lower()
        if text in self.numbers: return self.numbers[text]
        try: return float(text)
        except: return 0.0

    def eval_simple(self, expression):
        # Parser muy simple para "operacion num1 con num2"
        # Ejemplo: "suma mil más trescientos" -> no parsea "trescientos" si no está en dict
        # Para la demo usaremos entradas simples que coincidan
        words = expression.lower().split()
        
        # Estrategia naive para demo: buscar operador y dos números
        op = None
        nums = []
        
        for w in words:
            if w in self.operators:
                op = self.operators[w]
            val = self.parse_number(w)
            if val != 0 or w == 'cero':
                nums.append(val)
                
        if op and len(nums) >= 2:
            return op(nums[0], nums[1])
        return 0

def demo_unification():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   CMFO GRAND UNIFICATION: ESPAÑOL → FRACTAL → FÍSICA     ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # 1. CAPA DE INTENCIÓN
    print("1. INTENCIÓN (Álgebra de Español)")
    spanish = SpanishAlgebraCorrected()
    pregunta = "suma mil más veinte" # 1000 + 20
    
    val = spanish.eval_simple(pregunta)
    print(f"   Entrada: '{pregunta}'")
    print(f"   Interpretación CMFO: {val}")

    # 2. CAPA DE DECISIÓN
    print("\n2. GEOMETRÍA (Álgebra Fractal)")
    # Simulamos un vector de estado donde la "opción correcta" tiene la amplitud derivada
    state = [val * 0.1, val, val * 0.2]
    decision = phi_decision(state) # Debería elegir el índice 1 (valor más alto)
    print(f"   Vector de Estado: {state}")
    print(f"   Decisión Determinista: Índice {decision}")

    # 3. CAPA FÍSICA
    print("\n3. FÍSICA (Circuitos Reversibles)")
    # Hash del valor de intención
    header = int(val).to_bytes(4, 'big') * 20
    hash_rev, mem = sha256d_forward_reversible(header, trace=True)
    
    print(f"   Ejecutando Circuito Reversible...")
    if mem and mem.log:
        last = mem.log[-1]
        print(f"   Estado Final (Ronda {last[0]}):")
        print(f"   Carga Topológica: {last[6]:.4f}")
        print(f"   Carga No-Lineal:  {last[7]:.4f}")
        
    print("\n✓ UNIFICACIÓN COMPLETADA: Lenguaje -> Lógica -> Física")

if __name__ == "__main__":
    demo_unification()
