#!/usr/bin/env python3
"""
Demo: Spanish Algebra - Natural Language Interface for CMFO

This script demonstrates how Spanish natural language expressions
can be compiled to CMFO operators and executed deterministically.

Author: CMFO Research Team
License: MIT
"""

import cmfo
import re
from typing import Any, Dict, Tuple


class SpanishAlgebra:
    """
    Simplified Spanish Algebra compiler for demonstration purposes.
    
    Compiles Spanish mathematical expressions to CMFO operations.
    """
    
    def __init__(self):
        # Mapeo de palabras españolas a operadores CMFO
        self.operators = {
            # Operadores aritméticos
            'suma': cmfo.phi_add,
            'más': cmfo.phi_add,
            'agregar': cmfo.phi_add,
            'resta': cmfo.phi_sub,
            'menos': cmfo.phi_sub,
            'quitar': cmfo.phi_sub,
            'multiplica': cmfo.tensor_mul,
            'por': cmfo.tensor_mul,
            'veces': cmfo.tensor_mul,
            'divide': cmfo.tensor_div,
            'entre': cmfo.tensor_div,
        }
        
        # Modificadores cuantitativos
        self.modifiers = {
            'doble': lambda x: cmfo.tensor_mul(2, x),
            'duplo': lambda x: cmfo.tensor_mul(2, x),
            'triple': lambda x: cmfo.tensor_mul(3, x),
            'cuádruple': lambda x: cmfo.tensor_mul(4, x),
            'mitad': lambda x: cmfo.tensor_div(x, 2),
            'tercio': lambda x: cmfo.tensor_div(x, 3),
            'cuadrado': lambda x: cmfo.tensor_mul(x, x),
            'cubo': lambda x: cmfo.tensor_mul(cmfo.tensor_mul(x, x), x),
        }
        
        # Funciones especiales
        self.functions = {
            'raíz': cmfo.phi_sqrt,
            'raíz_cuadrada': cmfo.phi_sqrt,
        }
        
        # Números en español (para demostración)
        self.numbers = {
            'cero': 0, 'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4,
            'cinco': 5, 'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9,
            'diez': 10, 'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14,
            'quince': 15, 'dieciséis': 16, 'veinte': 20, 'treinta': 30,
            'cuarenta': 40, 'cincuenta': 50, 'cien': 100, 'mil': 1000,
        }
    
    def parse_number(self, text: str) -> float:
        """Extrae número de texto (español o dígitos)."""
        text = text.strip().lower()
        
        # Intenta número directo
        try:
            return float(text)
        except ValueError:
            pass
        
        # Busca en diccionario de números
        if text in self.numbers:
            return float(self.numbers[text])
        
        raise ValueError(f"No se puede interpretar '{text}' como número")
    
    def eval_simple(self, expression: str) -> Any:
        """
        Evalúa expresiones simples en español.
        
        Ejemplos:
        - "suma cinco más tres"
        - "el doble de diez"
        - "raíz cuadrada de dieciséis"
        """
        expression = expression.lower().strip()
        
        # Patrón: "operador número1 más/y número2"
        pattern_binary = r'(suma|resta|multiplica|divide)\s+(\w+)\s+(más|y|menos|por|entre)\s+(\w+)'
        match = re.search(pattern_binary, expression)
        if match:
            op_word = match.group(1)
            num1_word = match.group(2)
            num2_word = match.group(4)
            
            operator = self.operators.get(op_word)
            if operator:
                num1 = self.parse_number(num1_word)
                num2 = self.parse_number(num2_word)
                result = operator(num1, num2)
                return self._extract_real(result)
        
        # Patrón: "modificador de número"
        pattern_modifier = r'(doble|triple|mitad|cuadrado|cubo)\s+de\s+(\w+)'
        match = re.search(pattern_modifier, expression)
        if match:
            modifier_word = match.group(1)
            num_word = match.group(2)
            
            modifier = self.modifiers.get(modifier_word)
            if modifier:
                num = self.parse_number(num_word)
                result = modifier(num)
                return self._extract_real(result)
        
        # Patrón: "raíz cuadrada de número"
        pattern_sqrt = r'raíz\s+(cuadrada\s+)?de\s+(\w+)'
        match = re.search(pattern_sqrt, expression)
        if match:
            num_word = match.group(2)
            num = self.parse_number(num_word)
            result = cmfo.phi_sqrt(num)
            return self._extract_real(result)
        
        raise ValueError(f"No se puede interpretar la expresión: '{expression}'")
    
    def _extract_real(self, value: Any) -> float:
        """Extrae parte real de números complejos si es necesario."""
        if hasattr(value, 'real'):
            return float(value.real)
        return float(value)


def demo_basic_operations():
    """Demuestra operaciones básicas."""
    print("=" * 60)
    print("DEMO 1: Operaciones Aritméticas Básicas")
    print("=" * 60)
    
    algebra = SpanishAlgebra()
    
    examples = [
        "suma cinco más tres",
        "resta diez menos cuatro",
        "multiplica seis por siete",
        "divide veinte entre cuatro",
    ]
    
    for expr in examples:
        try:
            result = algebra.eval_simple(expr)
            print(f"📝 '{expr}'")
            print(f"✅ Resultado: {result}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


def demo_modifiers():
    """Demuestra modificadores cuantitativos."""
    print("=" * 60)
    print("DEMO 2: Modificadores Cuantitativos")
    print("=" * 60)
    
    algebra = SpanishAlgebra()
    
    examples = [
        "el doble de cinco",
        "el triple de tres",
        "la mitad de veinte",
        "el cuadrado de cuatro",
        "el cubo de dos",
    ]
    
    for expr in examples:
        try:
            result = algebra.eval_simple(expr)
            print(f"📝 '{expr}'")
            print(f"✅ Resultado: {result}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


def demo_functions():
    """Demuestra funciones especiales."""
    print("=" * 60)
    print("DEMO 3: Funciones Especiales")
    print("=" * 60)
    
    algebra = SpanishAlgebra()
    
    examples = [
        "raíz cuadrada de dieciséis",
        "raíz de cien",
        "raíz cuadrada de nueve",
    ]
    
    for expr in examples:
        try:
            result = algebra.eval_simple(expr)
            print(f"📝 '{expr}'")
            print(f"✅ Resultado: {result}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


def demo_determinism():
    """Demuestra el determinismo absoluto."""
    print("=" * 60)
    print("DEMO 4: Determinismo Absoluto")
    print("=" * 60)
    
    algebra = SpanishAlgebra()
    expression = "el doble de cinco"
    
    print(f"Ejecutando '{expression}' 5 veces...\n")
    
    results = []
    for i in range(5):
        result = algebra.eval_simple(expression)
        results.append(result)
        print(f"Ejecución {i+1}: {result}")
    
    # Verificar que todos son idénticos
    all_same = all(r == results[0] for r in results)
    print(f"\n{'✅' if all_same else '❌'} Determinismo: {'VERIFICADO' if all_same else 'FALLIDO'}")
    print(f"Todos los resultados son idénticos: {all_same}")


def demo_comparison_with_traditional():
    """Compara con sistemas tradicionales."""
    print("=" * 60)
    print("DEMO 5: Comparación con Python Tradicional")
    print("=" * 60)
    
    algebra = SpanishAlgebra()
    
    # Expresión en español
    spanish_expr = "suma cinco más tres"
    cmfo_result = algebra.eval_simple(spanish_expr)
    
    # Equivalente en Python tradicional
    python_result = 5 + 3
    
    print(f"Español Natural: '{spanish_expr}'")
    print(f"CMFO Resultado: {cmfo_result}")
    print(f"\nPython Tradicional: '5 + 3'")
    print(f"Python Resultado: {python_result}")
    print(f"\n{'✅' if abs(cmfo_result - python_result) < 0.001 else '❌'} Resultados coinciden")


def demo_interactive_mode():
    """Modo interactivo simple."""
    print("=" * 60)
    print("DEMO 6: Modo Interactivo")
    print("=" * 60)
    print("Escribe expresiones en español (o 'salir' para terminar)")
    print("Ejemplos:")
    print("  - suma cinco más tres")
    print("  - el doble de diez")
    print("  - raíz cuadrada de dieciséis")
    print("-" * 60)
    
    algebra = SpanishAlgebra()
    
    # Para demo, solo mostramos cómo funcionaría
    demo_inputs = [
        "suma dos más dos",
        "el triple de cinco",
        "raíz de nueve",
    ]
    
    for user_input in demo_inputs:
        print(f"\n>>> {user_input}")
        try:
            result = algebra.eval_simple(user_input)
            print(f"{result}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Ejecuta todas las demostraciones."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  CMFO Spanish Algebra - Demostración Interactiva".center(58) + "║")
    print("║" + "  Álgebra de Español: Lenguaje Natural → Matemática Exacta".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    demo_basic_operations()
    print("\n")
    
    demo_modifiers()
    print("\n")
    
    demo_functions()
    print("\n")
    
    demo_determinism()
    print("\n")
    
    demo_comparison_with_traditional()
    print("\n")
    
    demo_interactive_mode()
    print("\n")
    
    print("=" * 60)
    print("CONCLUSIÓN")
    print("=" * 60)
    print("✅ El Álgebra de Español permite expresar operaciones")
    print("   matemáticas en lenguaje natural español")
    print("✅ Compilación determinista: misma entrada → mismo resultado")
    print("✅ Sin ambigüedad: cada construcción mapea a un operador único")
    print("✅ Extensible: fácil agregar nuevas palabras y construcciones")
    print("\n📚 Ver docs/theory/SPANISH_ALGEBRA_SPEC.md para más detalles")
    print("=" * 60)


if __name__ == "__main__":
    main()
