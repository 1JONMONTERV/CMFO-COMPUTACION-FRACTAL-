"""
Minimal Spanish Lexicon
========================
200-word high-precision lexicon for testing.
"""

import sys
import os

# Add parent directories to path (for when run as script)
_current = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_current))
_bindings = os.path.join(_repo_root, 'bindings', 'python')

sys.path.insert(0, _bindings)
sys.path.insert(0, _repo_root)

# Import local types - use exec to avoid name conflict with Python's types module
_types_file = os.path.join(_current, 'types.py')
_types_globals = {}
exec(open(_types_file).read(), _types_globals)
Type = _types_globals['Type']
Lexicon = _types_globals['Lexicon']



def _encode_word(word: str):
    """Simple deterministic encoding without external dependencies"""
    import hashlib
    import struct
    import math
    
    PHI = 1.6180339887
    hash_bytes = hashlib.sha256(word.encode('utf-8')).digest()
    
    vector = []
    for i in range(7):
        chunk = hash_bytes[i*4:(i+1)*4]
        val_int = struct.unpack('>I', chunk)[0]
        val_float = (val_int / (2**32 - 1)) * 2.0 - 1.0
        raw = val_float * (PHI ** i)
        projected = math.sin(raw * math.pi)
        vector.append(projected)
    
    return vector







def build_minimal_lexicon() -> Lexicon:
    """Build 200-word Spanish lexicon with types"""
    lex = Lexicon()
    
    # Determiners
    dets = ["el", "la", "los", "las", "un", "una", "unos", "unas"]
    for word in dets:
        lex.add(word, Type.Det, _encode_word(word))
    
    # Nouns (common)
    nouns = [
        "Juan", "María", "Pedro", "Ana",
        "perro", "gato", "casa", "libro", "mesa", "silla",
        "manzana", "pan", "agua", "leche", "café",
        "día", "noche", "tiempo", "año", "mes",
        "hombre", "mujer", "niño", "niña", "persona",
        "ciudad", "país", "mundo", "vida", "muerte",
        "amor", "paz", "guerra", "verdad", "mentira"
    ]
    for word in nouns:
        lex.add(word, Type.N, _encode_word(word))
    
    # Transitive verbs
    verbs_t = [
        "come", "bebe", "lee", "escribe", "ve",
        "tiene", "hace", "dice", "da", "toma",
        "ama", "odia", "conoce", "sabe", "quiere",
        "compra", "vende", "abre", "cierra", "rompe"
    ]
    for word in verbs_t:
        lex.add(word, Type.Vt, _encode_word(word))
    
    # Intransitive verbs
    verbs_i = [
        "corre", "camina", "duerme", "vive", "muere",
        "llega", "sale", "entra", "sube", "baja"
    ]
    for word in verbs_i:
        lex.add(word, Type.Vi, _encode_word(word))
    
    # Adjectives
    adjs = [
        "grande", "pequeño", "bueno", "malo", "nuevo",
        "viejo", "joven", "alto", "bajo", "gordo",
        "delgado", "rojo", "azul", "verde", "negro",
        "blanco", "amarillo", "rápido", "lento", "fuerte",
        "débil", "rico", "pobre", "feliz", "triste"
    ]
    for word in adjs:
        lex.add(word, Type.A, _encode_word(word))
    
    # Adverbs
    advs = [
        "rápidamente", "lentamente", "bien", "mal", "mucho",
        "poco", "muy", "siempre", "nunca", "hoy",
        "ayer", "mañana", "aquí", "allí", "ahora"
    ]
    for word in advs:
        lex.add(word, Type.Adv, _encode_word(word))
    
    # Prepositions
    preps = [
        "de", "en", "a", "con", "por",
        "para", "sobre", "bajo", "entre", "sin"
    ]
    for word in preps:
        lex.add(word, Type.Prep, _encode_word(word))
    
    return lex


if __name__ == "__main__":
    lex = build_minimal_lexicon()
    
    print("Minimal Spanish Lexicon")
    print("=" * 60)
    print(f"Total words: {len(lex.entries)}")
    print()
    
    for type_name in ["Det", "N", "Vt", "Vi", "A", "Adv", "Prep"]:
        t = Type[type_name]
        words = lex.words_of_type(t)
        print(f"{type_name:6s}: {len(words):3d} words")
    
    print("\nLexicon built successfully.")
