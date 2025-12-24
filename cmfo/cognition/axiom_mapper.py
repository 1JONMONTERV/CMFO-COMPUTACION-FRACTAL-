"""
CMFO D13: Axiom Mapper
======================
The Translator from "Human Formalism" (LaTeX) to "Fractal Algebra" (7D Vectors).

Role:
- Takes a raw definition text (e.g., "Let X be a compact Hausdorff space")
- Extracts key terms using the Dictionary (D9).
- Constructs a candidate algebraic vector.
- Validates consistency.
"""

from typing import List, Dict, Tuple, Optional
try:
    from cmfo.semantics.algebra import SemanticAlgebra
    from cmfo.storage.bridge import KnowledgeBridge
except ImportError:
    pass

class AxiomMapper:
    # Scientific Translation Layer (English -> Spanish Axioms)
    # This allows extracting meaning from English papers using the Spanish Fractal Dictionary.
    TRANSLATION = {
        "set": "conjunto",
        "group": "grupo",
        "space": "espacio",
        "field": "campo",
        "function": "función",
        "map": "mapa",
        "mapping": "mapeo",
        "vector": "vector",
        "matrix": "matriz",
        "algebra": "álgebra",
        "system": "sistema",
        "theory": "teoría",
        "model": "modelo",
        "data": "datos",
        "information": "información",
        "value": "valor",
        "variable": "variable",
        "element": "elemento",
        "structure": "estructura",
        "point": "punto",
        "continuous": "continuo",
        "discrete": "discreto"
    }

    def __init__(self, bridge=None):
        self.bridge = bridge # D10 Bridge to lookup existing terms
        
    def formal_to_vector(self, definition_text: str) -> Optional[List[float]]:
        """
        Converts a formal definition string into a vector.
        """
        words = self._extract_keywords(definition_text)
        if not words:
            return None
            
        vectors = []
        for w in words:
            # 1. Try Translation
            term_es = self.TRANSLATION.get(w, w)
            
            # 2. Lookup in D9 Bridge (Spanish)
            if self.bridge:
                v = self.bridge.get_vector(term_es)
                if v:
                    vectors.append(list(v))
            else:
                pass
                
        if not vectors:
            return None
            
        # Algebra: Composition (Addition in toroidal space)
        # We start with ZERO and add
        result = [0.0] * 7 # 7D Vector
        
        # We need the Algebra logic. Ideally we import SemanticAlgebra
        # But for now, simple addition stub to demonstrate flow
        # In reality, we call SemanticAlgebra.compose(prop_vectors)
        
        # Stub sum
        for v in vectors:
            for i in range(7):
                result[i] += v[i]
                
        # Normalize? D8 Algebra handles normalization.
        
        return result

    def _extract_keywords(self, text: str) -> List[str]:
        """Simple NLP-free keyword extraction"""
        # Remove LaTeX commands
        import re
        clean = re.sub(r"\\[a-zA-Z]+", "", text)
        clean = re.sub(r"[^a-zA-Z ]", "", clean)
        
        stops = {"let", "be", "a", "an", "the", "we", "define", "is", "of", "to", "in", "and", "with"}
        tokens = clean.lower().split()
        keywords = [t for t in tokens if t not in stops and len(t) > 2]
        return keywords
