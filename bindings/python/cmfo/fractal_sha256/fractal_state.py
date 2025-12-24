"""
Fractal State Definition
=========================

Implements the 1024-position fractal state for reversible SHA-256.
"""

from typing import List, Set, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class FractalCell:
    """
    Single cell in fractal state representing one bit/qubit.
    
    Attributes:
        value: Boolean value (True/False or 1/0)
        origin: Set of input bit indices that contributed to this value
        round: Last modification round number
        operation: Operation that created this value
    """
    value: bool = False
    origin: Set[int] = field(default_factory=set)
    round: int = -1
    operation: str = "INIT"
    
    def copy(self) -> 'FractalCell':
        """Create a deep copy of the cell"""
        return FractalCell(
            value=self.value,
            origin=self.origin.copy(),
            round=self.round,
            operation=self.operation
        )

class FractalState:
    """
    1024-position fractal state for SHA-256d.
    
    Layout:
      [0-511]:   Working state (512 bits)
                 - [0-255]:   H state (8 words x 32 bits)
                 - [256-511]: Message schedule W
      [512-1023]: Ancillas and traceability (512 bits)
                 - [512-767]: Reversible operation ancillas
                 - [768-1023]: Reserved / Advanced tracing
    """
    
    # Constants
    SIZE = 1024
    WORK_START = 0
    WORK_END = 512
    ANCILLA_START = 512
    
    def __init__(self):
        self.cells: List[FractalCell] = [FractalCell() for _ in range(self.SIZE)]
        self.current_round = 0
    
    def reset(self):
        """Reset state to initial zeros"""
        self.cells = [FractalCell() for _ in range(self.SIZE)]
        self.current_round = 0
        
    def load_word(self, start_pos: int, word: int, origin_offset: int = -1):
        """
        Load 32-bit word into positions [start_pos : start_pos+32].
        
        Args:
            start_pos: Starting cell index
            word: 32-bit integer to load
            origin_offset: If >= 0, sets origin trace to {origin_offset + bit_idx}
        """
        if start_pos < 0 or start_pos + 32 > self.SIZE:
            raise IndexError(f"Word load out of bounds: {start_pos}")
            
        for i in range(32):
            # Extract i-th bit (big-endian loading usually, but SHA-256 internals allow
            # treating as array of bits. We'll map bit 31 (MSB) to index 0 for big-endian memory)
            # Actually, standard is: word is 32 bits. 
            # Let's align with big-endian: cell[0] is MSB, cell[31] is LSB.
            bit_val = (word >> (31 - i)) & 1 == 1
            
            self.cells[start_pos + i].value = bit_val
            self.cells[start_pos + i].round = self.current_round
            self.cells[start_pos + i].operation = "LOAD"
            
            if origin_offset >= 0:
                self.cells[start_pos + i].origin = {origin_offset + i}
            else:
                self.cells[start_pos + i].origin = set()
                
    def extract_word(self, start_pos: int) -> int:
        """
        Extract 32-bit word from positions [start_pos : start_pos+32].
        Assumes big-endian mapping (cell[0] is MSB).
        """
        if start_pos < 0 or start_pos + 32 > self.SIZE:
            raise IndexError(f"Word extract out of bounds: {start_pos}")
            
        word = 0
        for i in range(32):
            if self.cells[start_pos + i].value:
                word |= (1 << (31 - i))
        return word
    
    def get_trace(self, round_num: int) -> Dict[str, Any]:
        """Get all modifications in a specific round"""
        trace = {
            'round': round_num,
            'modified_cells': [],
            'operations': set()
        }
        
        for i, cell in enumerate(self.cells):
            if cell.round == round_num:
                trace['modified_cells'].append({
                    'position': i,
                    'value': cell.value,
                    'origin': list(cell.origin), # Convert set to list for JSON serialization
                    'operation': cell.operation
                })
                trace['operations'].add(cell.operation)
        
        trace['operations'] = list(trace['operations'])
        return trace
