"""
CMFO Shift-Reduce Parser
=========================
Deterministic typed parser with lookahead and priority.

Criteria:
1. Determinism: same input → same tree → same state
2. Lookahead: resolve N Vt N vs N Vi
3. Priority: DET > MOD_N > REL > APP_O > ADV_V > APP_S > TENSE > NEG
4. Type checking: no invalid op(x,y)
5. Single output: len(stack)==1 ⇒ S valid
"""

import sys
import os
from typing import List, Optional, Tuple

# Setup paths
_current = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_current))
_bindings = os.path.join(_repo_root, 'bindings', 'python')

sys.path.insert(0, _bindings)
sys.path.insert(0, _repo_root)

# Load local modules using exec to avoid import conflicts
_ops_file = os.path.join(_current, 'operators.py')
_types_file = os.path.join(_current, 'types.py')

_ops_globals = {}
exec(open(_ops_file).read(), _ops_globals)
OPERATORS = _ops_globals['OPERATORS']

_types_globals = {}
exec(open(_types_file).read(), _types_globals)
Type = _types_globals['Type']
TypedVector = _types_globals['TypedVector']
TypeChecker = _types_globals['TypeChecker']
Lexicon = _types_globals['Lexicon']



class ParseError(Exception):
    """Parsing failed"""
    pass


class Parser:
    """Shift-reduce parser with type checking"""
    
    # Operator priority (higher = reduce first)
    PRIORITY = {
        'DET': 9,
        'MOD_N': 8,
        'REL': 7,
        'APP_O': 6,
        'ADV_V': 5,
        'APP_S': 4,
        'TENSE': 3,
        'NEG_SCOPE': 2,
        'COMP': 1
    }
    
    def __init__(self, lexicon: Lexicon):
        self.lexicon = lexicon
        self.stack: List[TypedVector] = []
        self.input_buffer: List[str] = []
        
    def parse(self, words: List[str]) -> TypedVector:
        """
        Parse sentence to final state.
        
        Returns:
            TypedVector with type S
            
        Raises:
            ParseError if parsing fails
        """
        # Reset
        self.stack = []
        self.input_buffer = words[:]
        
        # Shift-reduce loop
        while self.input_buffer or len(self.stack) > 1:
            # Try reduce first (priority-based)
            if self._try_reduce():
                continue
            
            # Otherwise shift
            if self.input_buffer:
                self._shift()
            else:
                # No more input and can't reduce
                raise ParseError(f"Parse incomplete: stack={[s.word for s in self.stack]}")
        
        # Check final state
        if len(self.stack) != 1:
            raise ParseError(f"Parse failed: stack size = {len(self.stack)}")
        
        final = self.stack[0]
        if final.type != Type.S:
            raise ParseError(f"Parse produced non-sentence: {final.type}")
        
        return final
    
    def _shift(self):
        """Shift next word from input to stack"""
        word = self.input_buffer.pop(0)
        entry = self.lexicon.lookup(word)
        
        if entry is None:
            raise ParseError(f"Unknown word: {word}")
        
        self.stack.append(entry)
    
    def _try_reduce(self) -> bool:
        """
        Try to reduce top of stack.
        Returns True if reduction happened.
        """
        # Try binary operators (need at least 2 items)
        if len(self.stack) >= 2:
            # Check all binary operators by priority
            for op_name in sorted(self.PRIORITY.keys(), key=lambda x: -self.PRIORITY[x]):
                if op_name in ['NEG_SCOPE', 'TENSE']:
                    continue  # Unary
                
                if self._try_binary_reduce(op_name):
                    return True
        
        # Try unary operators (need at least 1 item)
        if len(self.stack) >= 1:
            for op_name in ['NEG_SCOPE', 'TENSE']:
                if self._try_unary_reduce(op_name):
                    return True
        
        return False
    
    def _try_binary_reduce(self, op_name: str) -> bool:
        """Try to apply binary operator to top 2 stack items"""
        if len(self.stack) < 2:
            return False
        
        # Get top 2
        right = self.stack[-1]
        left = self.stack[-2]
        
        # Check type compatibility
        input_types = (left.type, right.type)
        output_type = TypeChecker.check(op_name, input_types)
        
        if output_type is None:
            return False  # Type error, can't reduce
        
        # Lookahead check for APP_O vs APP_S disambiguation
        if op_name == 'APP_O':
            # Only reduce if we have object, not subject
            # Heuristic: if next word is noun, this might be subject
            if self.input_buffer and self.lexicon.get_type(self.input_buffer[0]) in [Type.N, Type.N_det]:
                # Don't reduce yet, might be APP_S later
                return False
        
        # Apply operator
        op_func = OPERATORS[op_name]['func']
        result_vec = op_func(left.vector, right.vector)
        
        # Create result
        result = TypedVector(
            vector=result_vec,
            type=output_type,
            word=f"({left.word} {op_name} {right.word})"
        )
        
        # Replace top 2 with result
        self.stack.pop()
        self.stack.pop()
        self.stack.append(result)
        
        return True
    
    def _try_unary_reduce(self, op_name: str) -> bool:
        """Try to apply unary operator to top stack item"""
        if len(self.stack) < 1:
            return False
        
        top = self.stack[-1]
        
        # Check type
        input_types = (top.type,)
        output_type = TypeChecker.check(op_name, input_types)
        
        if output_type is None:
            return False
        
        # Apply operator
        op_func = OPERATORS[op_name]['func']
        result_vec = op_func(top.vector)
        
        # Create result
        result = TypedVector(
            vector=result_vec,
            type=output_type,
            word=f"({op_name} {top.word})"
        )
        
        # Replace top with result
        self.stack.pop()
        self.stack.append(result)
        
        return True


if __name__ == "__main__":
    # Quick test
    from cortex.encoder import FractalEncoder
    
    print("CMFO Parser")
    print("=" * 60)
    
    # Create minimal lexicon
    encoder = FractalEncoder()
    lex = Lexicon()
    
    # Add test words
    lex.add("Juan", Type.N, encoder.encode("Juan").v)
    lex.add("come", Type.Vt, encoder.encode("come").v)
    lex.add("manzana", Type.N, encoder.encode("manzana").v)
    
    # Parse
    parser = Parser(lex)
    
    try:
        result = parser.parse(["Juan", "come", "manzana"])
        print(f"\n[OK] Parsed: Juan come manzana")
        print(f"     Type: {result.type}")
        print(f"     Vector: {result.vector[:3]}...")
    except ParseError as e:
        print(f"\n[FAIL] {e}")
    
    print("\nParser loaded successfully.")
