"""
CMFO Type System
================
Grammatical type checking and validation
"""

from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass


class Type(Enum):
    """Grammatical types"""
    # Basic types
    N = "N"          # Noun
    V = "V"          # Verb (generic)
    Vt = "Vt"        # Transitive verb
    Vi = "Vi"        # Intransitive verb / Verb with object
    A = "A"          # Adjective
    Adv = "Adv"      # Adverb
    Det = "Det"      # Determiner
    Prep = "Prep"    # Preposition
    
    # Derived types
    N_det = "N_det"  # Determined noun
    N_rel = "N_rel"  # Relational noun
    PP = "PP"        # Prepositional phrase
    S = "S"          # Sentence
    
    # Special
    UNKNOWN = "?"


@dataclass
class TypedVector:
    """Vector with grammatical type"""
    vector: List[float]
    type: Type
    word: Optional[str] = None


class TypeChecker:
    """Type checking for grammatical operations"""
    
    # Type transition table: (op_name, input_types) -> output_type
    TYPE_RULES = {
        ('DET', (Type.Det, Type.N)): Type.N_det,
        ('DET', (Type.Det, Type.N_det)): Type.N_det,  # Can re-determine
        
        ('MOD_N', (Type.N, Type.A)): Type.N,
        ('MOD_N', (Type.N_det, Type.A)): Type.N_det,
        
        ('REL', (Type.N, Type.PP)): Type.N_rel,
        ('REL', (Type.N_det, Type.PP)): Type.N_rel,
        
        ('APP_O', (Type.Vt, Type.N)): Type.Vi,
        ('APP_O', (Type.Vt, Type.N_det)): Type.Vi,
        ('APP_O', (Type.Vt, Type.N_rel)): Type.Vi,
        
        ('APP_S', (Type.Vi, Type.N)): Type.S,
        ('APP_S', (Type.Vi, Type.N_det)): Type.S,
        ('APP_S', (Type.Vi, Type.N_rel)): Type.S,
        
        ('ADV_V', (Type.V, Type.Adv)): Type.V,
        ('ADV_V', (Type.Vt, Type.Adv)): Type.Vt,
        ('ADV_V', (Type.Vi, Type.Adv)): Type.Vi,
        
        ('NEG_SCOPE', (Type.S,)): Type.S,
        
        ('TENSE', (Type.S,)): Type.S,
        
        ('COMP', (Type.A, Type.N, Type.N)): Type.S,
        ('COMP', (Type.A, Type.N_det, Type.N_det)): Type.S,
    }
    
    @staticmethod
    def check(op_name: str, input_types: Tuple[Type, ...]) -> Optional[Type]:
        """
        Check if operation is type-safe.
        
        Returns:
            Output type if valid, None if type error
        """
        key = (op_name, input_types)
        return TypeChecker.TYPE_RULES.get(key)
    
    @staticmethod
    def is_valid(op_name: str, input_types: Tuple[Type, ...]) -> bool:
        """Check if operation is type-safe (boolean)"""
        return TypeChecker.check(op_name, input_types) is not None


class Lexicon:
    """Minimal lexicon with type annotations"""
    
    def __init__(self):
        self.entries = {}
    
    def add(self, word: str, type: Type, vector: List[float]):
        """Add word to lexicon"""
        self.entries[word] = TypedVector(vector, type, word)
    
    def lookup(self, word: str) -> Optional[TypedVector]:
        """Lookup word in lexicon"""
        return self.entries.get(word)
    
    def get_type(self, word: str) -> Type:
        """Get type of word"""
        entry = self.lookup(word)
        return entry.type if entry else Type.UNKNOWN
    
    def has_word(self, word: str) -> bool:
        """Check if word exists"""
        return word in self.entries
    
    def words_of_type(self, type: Type) -> List[str]:
        """Get all words of given type"""
        return [word for word, entry in self.entries.items() if entry.type == type]


if __name__ == "__main__":
    # Test type checker
    print("CMFO Type System")
    print("=" * 60)
    
    # Test valid operations
    tests = [
        ('DET', (Type.Det, Type.N), Type.N_det),
        ('APP_O', (Type.Vt, Type.N), Type.Vi),
        ('APP_S', (Type.Vi, Type.N), Type.S),
        ('NEG_SCOPE', (Type.S,), Type.S),
    ]
    
    print("\nType Checking Tests:")
    for op, inputs, expected in tests:
        result = TypeChecker.check(op, inputs)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"  {status} {op}{inputs} -> {result}")
    
    print("\nType system loaded successfully.")
