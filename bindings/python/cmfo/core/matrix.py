import ctypes
import numpy as np
from .native_lib import NativeLib

class T7Matrix:
    """
    High-Performance Wrapper around C++ Matrix7x7 Engine.
    """
    def __init__(self, _ptr=None):
        self.lib = NativeLib.get()
        if _ptr:
            self.ptr = _ptr
            self.owned = False
        else:
            self.ptr = self.lib.Matrix7x7_Create()
            self.owned = True

    def __del__(self):
        if hasattr(self, 'owned') and self.owned and hasattr(self, 'ptr') and self.ptr:
             self.lib.Matrix7x7_Destroy(self.ptr)

    @staticmethod
    def identity():
        m = T7Matrix()
        m.lib.Matrix7x7_SetIdentity(m.ptr)
        return m

    def __matmul__(self, other):
        """Matrix Multiplication (A @ B)"""
        if isinstance(other, T7Matrix):
            result = T7Matrix()
            self.lib.Matrix7x7_Multiply(self.ptr, other.ptr, result.ptr)
            return result
        elif isinstance(other, (list, tuple, np.ndarray)):
            # Vector Application
            return self.apply(other)
        else:
            raise TypeError("Unsupported operand for @")

    def apply(self, vector):
        """Apply Matrix to a 7D Vector (Complex supported)"""
        # Prepare input
        v = np.array(vector, dtype=complex)
        if v.size != 7:
            raise ValueError("T7Matrix expects 7D vector")
        
        v_real = np.ascontiguousarray(v.real, dtype=np.float64)
        v_imag = np.ascontiguousarray(v.imag, dtype=np.float64)
        
        out_real = np.zeros(7, dtype=np.float64)
        out_imag = np.zeros(7, dtype=np.float64)
        
        # Pointers
        p_in_r = v_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_in_i = v_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_out_r = out_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_out_i = out_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.Matrix7x7_Apply(self.ptr, p_in_r, p_in_i, p_out_r, p_out_i)
        
        return out_real + 1j * out_imag

    def evolve_state(self, initial_state, steps=1):
        """
        High-Performance C++ Simulation Loop.
        Performs v = sin(Matrix @ v) for 'steps'.
        Modifies state in-place? No, returns new final state.
        """
        v = np.array(initial_state, dtype=complex)
        if v.size != 7:
            raise ValueError("State must be 7D")
            
        # We need mutable buffers for the C function to write back into
        # But wait, C function arg is 'vec_real', 'vec_imag'. 
        # Is it in/out? Yes, logic says load, loop, save back.
        
        v_real = np.ascontiguousarray(v.real, dtype=np.float64)
        v_imag = np.ascontiguousarray(v.imag, dtype=np.float64)
        
        p_real = v_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_imag = v_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.Matrix7x7_Evolve(self.ptr, p_real, p_imag, ctypes.c_int(steps))
        
        return v_real + 1j * v_imag
