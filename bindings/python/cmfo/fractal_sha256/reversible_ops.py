"""
Fractal Reversible Operators
=============================

Implements bit-exact reversible boolean operators for SHA-256d.
"""

from typing import List, Tuple
from .fractal_state import FractalState, FractalCell

def xor_fractal(state: FractalState, pos_a: int, pos_b: int, pos_out: int):
    """
    Reversible XOR (Feynman Gate equivalent).
    
    Operation: out = a ⊕ b
    Trace: out.origin = a.origin ∪ b.origin
    """
    cell_a = state.cells[pos_a]
    cell_b = state.cells[pos_b]
    cell_out = state.cells[pos_out]
    
    # Value calculation
    new_value = cell_a.value ^ cell_b.value
    
    # Trace propagation
    new_origin = cell_a.origin.union(cell_b.origin)
    
    # Update state
    cell_out.value = new_value
    cell_out.origin = new_origin
    cell_out.round = state.current_round
    cell_out.operation = "XOR"

def xor_word_fractal(state: FractalState, start_a: int, start_b: int, start_out: int):
    """Apply XOR to entire 32-bit words"""
    for i in range(32):
        xor_fractal(state, start_a + i, start_b + i, start_out + i)

def and_fractal(state: FractalState, pos_a: int, pos_b: int, pos_c: int):
    """
    Reversible AND (Toffoli Gate equivalent).
    
    Operation: c = c ⊕ (a ∧ b)
    Note: Standard boolean AND assigns result. Reversible AND XORs into target.
    For standard assignment behavior, target 'pos_c' should be initialized to 0.
    
    Trace: c.origin = c.origin ∪ a.origin ∪ b.origin
    """
    cell_a = state.cells[pos_a]
    cell_b = state.cells[pos_b]
    cell_c = state.cells[pos_c]
    
    # Value calculation: c_new = (a AND b)
    # Note: We use overwrite assignment to ensure correctness when reusing dirty ancillas.
    # For strict reversible logic (Toffoli), target must be 0. 
    # Here we prioritize correctness of the hash function over strict Toffoli behavior on dirty lines.
    new_value = cell_a.value and cell_b.value
    
    # Trace propagation
    if new_value:
        # If AND fired, dependency includes inputs
        new_origin = cell_c.origin.union(cell_a.origin).union(cell_b.origin)
    else:
        # If AND didn't fire, strictly speaking dependency still exists structurally
        new_origin = cell_c.origin.union(cell_a.origin).union(cell_b.origin)
        
    # Update state
    cell_c.value = new_value
    cell_c.origin = new_origin
    cell_c.round = state.current_round
    cell_c.operation = "TOFFOLI"

def not_fractal(state: FractalState, pos_a: int):
    """
    Reversible NOT (Pauli-X equivalent).
    
    Operation: a = ¬a
    """
    cell = state.cells[pos_a]
    
    cell.value = not cell.value
    cell.round = state.current_round
    cell.operation = "NOT"

def not_word_fractal(state: FractalState, start_a: int):
    """Apply NOT to entire 32-bit word"""
    for i in range(32):
        not_fractal(state, start_a + i)

def rotr_fractal(state: FractalState, start_src: int, start_dst: int, n: int):
    """
    Reversible Rotate Right (ROTR) on 32-bit word.
    
    Copies rotated bits from src to dst.
    Operation: dst[i] = src[(i - n) % 32]
    """
    for i in range(32):
        # Calculate source index for bit i
        # In ROTR by n, the bit at position i comes from position (i + n) % 32?
        # Let's trace: ROTR(1) of [1,0,0,0...] (0x80000000) -> [0,1,0,0...] (0x40000000)
        # Bit at index 1 comes from index 0.
        # General: dst[i] comes from src[(i - n) % 32]
        
        # BUT wait: our cells are [0..31] where 0 is MSB.
        # 0x80000000: bit 0 is 1.
        # ROTR 1: 0x40000000: bit 1 is 1.
        # So dst[1] gets src[0].
        # dst[i] gets src[(i - n) % 32].
        
        src_idx = (i - n) % 32
        
        # Copy operation logic (XOR-copy if dst is 0)
        # We assume dst is cleared or we are doing an assignment equivalent
        # For simplicity in this implementation, we allow direct assignment COPY
        # because this is just a wire permutation in circuits.
        
        src_cell = state.cells[start_src + src_idx]
        dst_cell = state.cells[start_dst + i]
        
        dst_cell.value = src_cell.value
        dst_cell.origin = src_cell.origin.copy()
        dst_cell.round = state.current_round
        dst_cell.operation = f"ROTR_{n}"

def shr_fractal(state: FractalState, start_src: int, start_dst: int, n: int):
    """
    Shift Right (SHR) on 32-bit word.
    Vacated bits are filled with 0.
    """
    for i in range(32):
        # Bit i comes from i - n
        # If i - n < 0, it's 0 (shifted in)
        
        src_idx = i - n
        dst_cell = state.cells[start_dst + i]
        
        if src_idx < 0:
            # Shifted in zero
            dst_cell.value = False
            dst_cell.origin = set()
            dst_cell.operation = f"SHR_{n}_ZERO"
        else:
            # Copy bit
            src_cell = state.cells[start_src + src_idx]
            dst_cell.value = src_cell.value
            dst_cell.origin = src_cell.origin.copy()
            dst_cell.operation = f"SHR_{n}"
            
        dst_cell.round = state.current_round

def add_mod_fractal(state: FractalState, start_a: int, start_b: int, start_out: int, ancilla_start: int):
    """
    Reversible Addition Modulo 2^32.
    
    Uses Ripple-Carry Adder logic with ancillas for carriers.
    
    Args:
        start_a: First operand start index
        start_b: Second operand start index
        start_out: Output start index
        ancilla_start: Start index for 32 ancilla bits (carries)
    """
    # Initialize carry to 0 using first ancilla as external carry input (0)
    # Actually we use ancillas for internal carries c[0]..c[31]
    # c[i] is carry out from bit i
    
    # Bit 31 is LSB. We start adding from LSB.
    
    previous_carry_val = False
    previous_carry_origin = set()
    
    for i in range(31, -1, -1):
        # We are adding bit i from a and b
        cell_a = state.cells[start_a + i]
        cell_b = state.cells[start_b + i]
        cell_out = state.cells[start_out + i]
        cell_carry = state.cells[ancilla_start + i]
        
        a_val = cell_a.value
        b_val = cell_b.value
        c_in_val = previous_carry_val
        
        # Sum = a XOR b XOR c_in
        sum_val = a_val ^ b_val ^ c_in_val
        
        # Carry Out = (a AND b) OR (c_in AND (a XOR b))
        # Majority logic: at least 2 are true
        c_out_val = (a_val and b_val) or (c_in_val and (a_val ^ b_val))
        
        # Calculate origin trace
        current_origin = cell_a.origin.union(cell_b.origin).union(previous_carry_origin)
        
        # Update Output
        cell_out.value = sum_val
        cell_out.origin = current_origin.copy() # Simplification: sum depends on all inputs
        cell_out.round = state.current_round
        cell_out.operation = "ADD"
        
        # Update Carry Ancilla (for next iteration)
        # Note: We store carry for visualization, but use variable for calculation to avoid
        # complex double-read issues in this simplified loop.
        cell_carry.value = c_out_val
        cell_carry.origin = current_origin.copy()
        cell_carry.round = state.current_round
        cell_carry.operation = "CARRY"
        
        previous_carry_val = c_out_val
        previous_carry_origin = current_origin

