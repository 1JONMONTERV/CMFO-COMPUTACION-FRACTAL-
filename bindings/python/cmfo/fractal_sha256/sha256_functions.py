"""
SHA-256 Fractal Functions
==========================

Implements Ch, Maj, Sigma0, Sigma1, sigma0, sigma1 using reversible operators.
"""

from .fractal_state import FractalState
from .reversible_ops import (
    xor_fractal, and_fractal, not_fractal, 
    rotr_fractal, shr_fractal, xor_word_fractal, not_word_fractal
)

def Ch_fractal(state: FractalState, e_pos: int, f_pos: int, g_pos: int, 
               out_pos: int, ancilla_pos: int):
    """
    Ch(e, f, g) = (e AND f) XOR ((NOT e) AND g)
    
    Fractal implementation:
    1. temp1 = e AND f (store in ancilla)
    2. temp2 = (NOT e) AND g (store in ancilla + 32)
    3. out = temp1 XOR temp2
    """
    # 1. temp1 = e AND f
    # We apply bitwise AND for each of the 32 bits
    for i in range(32):
        and_fractal(state, e_pos + i, f_pos + i, ancilla_pos + i)
        
    # 2. temp2 = (NOT e) AND g
    # We need 'NOT e'. Since we can't destructively modify 'e' (we need it later),
    # we can use logic: (A AND B) XOR (? AND C)
    # Actually, reversible logic allows using NOT gate, computing, then NOT gate again to restore.
    
    # Invert e (in place)
    not_word_fractal(state, e_pos)
    
    # Compute temp2 = e' AND g (where e' is NOT e)
    for i in range(32):
        and_fractal(state, e_pos + i, g_pos + i, ancilla_pos + 32 + i)
        
    # Restore e (invert back)
    not_word_fractal(state, e_pos)
    
    # 3. out = temp1 XOR temp2
    xor_word_fractal(state, ancilla_pos, ancilla_pos + 32, out_pos)
    
    # Cleanup ancillas? 
    # In strictly reversible computing we'd need to uncompute temp1/temp2.
    # For now, we assume ancillas are "dirty" and we use fresh ones or clear them later if needed.
    # To keep trace clean, we leave them.

def Maj_fractal(state: FractalState, a_pos: int, b_pos: int, c_pos: int,
                out_pos: int, ancilla_pos: int):
    """
    Maj(a, b, c) = (a AND b) XOR (a AND c) XOR (b AND c)
    """
    # 1. temp1 = a AND b (ancilla)
    for i in range(32):
        and_fractal(state, a_pos + i, b_pos + i, ancilla_pos + i)
        
    # 2. temp2 = a AND c (ancilla + 32)
    for i in range(32):
        and_fractal(state, a_pos + i, c_pos + i, ancilla_pos + 32 + i)
        
    # 3. temp3 = b AND c (ancilla + 64)
    for i in range(32):
        and_fractal(state, b_pos + i, c_pos + i, ancilla_pos + 64 + i)
        
    # 4. out = temp1 XOR temp2 XOR temp3
    xor_word_fractal(state, ancilla_pos, ancilla_pos + 32, out_pos)
    xor_word_fractal(state, ancilla_pos + 64, out_pos, out_pos) # XOR into result again? No, XOR is (A^B)^C

    # Wait, xor_word_fractal computes: dst = dst XOR src1 XOR src2? 
    # No, our def is: out = a XOR b.
    # So step 4:
    # out = temp1 XOR temp2
    # out = out XOR temp3
    
    # Correcting call:
    # First: out = temp1 XOR temp2 
    # Note: xor_word_fractal(state, a, b, out) sets out = a XOR b.
    xor_word_fractal(state, ancilla_pos, ancilla_pos + 32, out_pos)
    
    # Second: out = out XOR temp3
    # We can't use 'out' as input 'a' and output 'out' simultaneously if implementation doesn't support it.
    # Our xor_fractal implementation:
    # new_value = cell_a.value ^ cell_b.value
    # cell_out.value = new_value
    # This is safe even if cell_out is aliased to cell_a.
    
    # However, logic is: out_new = out_current XOR temp3
    # Our func is: out = a XOR b
    # So call: xor_word_fractal(state, out_pos, ancilla_pos + 64, out_pos)
    
    xor_word_fractal(state, out_pos, ancilla_pos + 64, out_pos)


def Sigma0_fractal(state: FractalState, x_pos: int, out_pos: int, ancilla_pos: int):
    """
    Σ₀(x) = ROTR²(x) XOR ROTR¹³(x) XOR ROTR²²(x)
    """
    # 1. temp1 = ROTR 2 (ancilla)
    rotr_fractal(state, x_pos, ancilla_pos, 2)
    
    # 2. temp2 = ROTR 13 (ancilla + 32)
    rotr_fractal(state, x_pos, ancilla_pos + 32, 13)
    
    # 3. temp3 = ROTR 22 (ancilla + 64)
    rotr_fractal(state, x_pos, ancilla_pos + 64, 22)
    
    # 4. out = temp1 XOR temp2 XOR temp3
    xor_word_fractal(state, ancilla_pos, ancilla_pos + 32, out_pos)
    xor_word_fractal(state, out_pos, ancilla_pos + 64, out_pos)

def Sigma1_fractal(state: FractalState, x_pos: int, out_pos: int, ancilla_pos: int):
    """
    Σ₁(x) = ROTR⁶(x) XOR ROTR¹¹(x) XOR ROTR²⁵(x)
    """
    rotr_fractal(state, x_pos, ancilla_pos, 6)
    rotr_fractal(state, x_pos, ancilla_pos + 32, 11)
    rotr_fractal(state, x_pos, ancilla_pos + 64, 25)
    
    xor_word_fractal(state, ancilla_pos, ancilla_pos + 32, out_pos)
    xor_word_fractal(state, out_pos, ancilla_pos + 64, out_pos)

def sigma0_fractal(state: FractalState, x_pos: int, out_pos: int, ancilla_pos: int):
    """
    σ₀(x) = ROTR⁷(x) XOR ROTR¹⁸(x) XOR SHR³(x)
    """
    rotr_fractal(state, x_pos, ancilla_pos, 7)
    rotr_fractal(state, x_pos, ancilla_pos + 32, 18)
    shr_fractal(state, x_pos, ancilla_pos + 64, 3)
    
    xor_word_fractal(state, ancilla_pos, ancilla_pos + 32, out_pos)
    xor_word_fractal(state, out_pos, ancilla_pos + 64, out_pos)

def sigma1_fractal(state: FractalState, x_pos: int, out_pos: int, ancilla_pos: int):
    """
    σ₁(x) = ROTR¹⁷(x) XOR ROTR¹⁹(x) XOR SHR¹⁰(x)
    """
    rotr_fractal(state, x_pos, ancilla_pos, 17)
    rotr_fractal(state, x_pos, ancilla_pos + 32, 19)
    shr_fractal(state, x_pos, ancilla_pos + 64, 10)
    
    xor_word_fractal(state, ancilla_pos, ancilla_pos + 32, out_pos)
    xor_word_fractal(state, out_pos, ancilla_pos + 64, out_pos)
