"""
CMFO D9: Semantic Compiler (Ingestion Engine)
=============================================
Transforms Raw Dictionary Definitions -> Algebraic Vectors.

Pipeline:
1. Parse Definition (Natural Language)
2. Extract Algebraic Components (Base + Modifiers)
3. Calculate Vector (D8 Algebra)
4. Commit to Storage (D9 Infrastructure)

Target: RAE-style definitions ("Sustantivo. Animal que...")
"""

import re
from typing import List, Dict, Optional, Tuple

# Internal dependencies
try:
    from ..semantics.algebra import SemanticAlgebra, PROPERTY_VECTORS, AXES
    from ..storage.core import lexicon, vectors, AlgebraicDef, VectorEntry
except ImportError:
    # Adjust for local execution context if needed
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from semantics.algebra import SemanticAlgebra, PROPERTY_VECTORS, AXES
    from storage.core import lexicon, vectors, AlgebraicDef, VectorEntry


class SemanticCompiler:
    """
    Compiles natural language definitions into 7D vectors.
    """
    
    def __init__(self, axiom_version: str = "CMFO-D8-v1"):
        self.axiom_version = axiom_version
        
        # Simple heuristic mapping for "Genus" (Base Types)
        # In a full system, this would be a large lookup table
        self.base_map = {
            "animal": ["entidad", "vivo", "animal"],
            "persona": ["entidad", "vivo", "humano"],
            "planta": ["entidad", "vivo"],
            "cosa": ["entidad"],
            "acción": ["acción"],
            "cualidad": ["orden"], # Qualities usually map to Order/Structure
            "sentimiento": ["mente", "conexión"],
            "lugar": ["entidad", "orden"],
            "tiempo": ["tiempo"]
        }
        
        # Modifiers map (Adjectives/Verbs to Properties)
        self.modifier_map = {
            "doméstico": ["fiel", "conexión"],
            "salvaje": ["acción", "caos"], # Chaos = high entropy/freedom
            "bueno": ["bien"],
            "malo": ["mal"],
            "grande": ["entidad"], # Magnitude
            "rápido": ["acción", "tiempo"],
            "verdadero": ["verdad"],
            "falso": ["mentira"]
        }
    
    def compile(self, term: str, definition_text: str, source: str = "manual") -> str:
        """
        Compile a term from its definition.
        Returns: Vector ID (hash)
        """
        # 1. Parse (Extract Components)
        components = self._parse_definition(definition_text)
        
        # 2. Synthesize Vector (Algebraic Composition)
        # We flatten the components list for D8 composition
        flat_props = []
        for c in components["base"] + components["properties"]:
            flat_props.append(c)
            
        # Calculate using D8 Algebra
        vector = SemanticAlgebra.compose(flat_props)
        
        # 3. Create Definition Object (CAF)
        def_obj = AlgebraicDef(
            term=term,
            algebraic_def={
                "base": components["base"],
                "properties": components["properties"],
                "raw_def": [definition_text] # Keep trace
            },
            source=source,
            lang="es"
        )
        
        # 4. Commit to Storage
        # A. Store Vector
        vid = vectors.add(vector, term, self.axiom_version)
        
        # B. Store Definition
        lexicon.add_entry(def_obj, vid)
        
        return vid
        
    def _parse_definition(self, text: str) -> Dict[str, List[str]]:
        """
        Heuristic Parser: Extracts Algebraic Atoms from Text.
        "Animal doméstico que ladra" -> Base=[animal], Props=[doméstico]
        """
        text = text.lower()
        words = re.findall(r'\w+', text)
        
        base = []
        props = []
        
        # 1. Identify Base (Genus)
        # Look for the first noun that matches our ontology
        for word in words:
            if word in self.base_map:
                base.extend(self.base_map[word])
                break # Usually only one Genus per definition
        
        # Default base if none found
        if not base:
            base = ["entidad"] # Fallback to generic Entity
            
        # 2. Identify Modifiers (Differentia)
        for word in words:
            if word in self.modifier_map:
                props.extend(self.modifier_map[word])
            elif word in PROPERTY_VECTORS and word not in base:
                # Direct match with D8 Ontology
                props.append(word)
                
        return {"base": base, "properties": props}


if __name__ == "__main__":
    print("CMFO D9 Semantic Compiler")
    print("=========================")
    
    compiler = SemanticCompiler()
    
    # Test Cases
    definitions = [
        ("perro", "Animal doméstico fiel"),
        ("lobo", "Animal salvaje"),
        ("mentira", "Acción falsa y mala"),
        ("verdad", "Cualidad verdadera"),
        ("robot", "Cosa con mente artificial") # 'artificial' not in map, might lose nuance
    ]
    
    for term, definition in definitions:
        print(f"\nCompiling '{term}': '{definition}'")
        vid = compiler.compile(term, definition)
        
        # Verify
        vec = vectors.get(vid)
        print(f" -> Vector ID: {vid[:16]}...")
        print(f" -> Vector: {[round(x, 2) for x in vec]}")
        
    print("\nCompilation Test Complete.")
