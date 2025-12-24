import sys
import ctypes
import numpy as np
from .native_lib import NativeLib
from .structural import FractalMatrix7, FractalVector7

class T7Matrix:
    """
    High-Performance Wrapper around C++ Matrix7x7 Engine.
    """
    def __init__(self, _ptr=None):
        self.lib = NativeLib.get()
        if not self.lib:
            self.ptr = None
            self.owned = False
            # Fallback Logic: Try NumPy first, then pure Structural
            try:
                import numpy as np
                self.python_matrix = np.zeros((7, 7), dtype=complex)
                self.backend = "numpy"
            except ImportError:
                self.python_matrix = FractalMatrix7()
                self.backend = "structural"
            return

        if _ptr:
            self.ptr = _ptr
            self.owned = False
        else:
            self.ptr = self.lib.Matrix7x7_Create()
            self.owned = True
            self.python_matrix = None
            self.backend = "native"

    def __del__(self):
        if hasattr(self, 'owned') and self.owned and hasattr(self, 'ptr') and self.ptr:
             self.lib.Matrix7x7_Destroy(self.ptr)

    @staticmethod
    def identity():
        m = T7Matrix()
        if not m.ptr:
             if m.backend == "numpy":
                 m.python_matrix = np.eye(7, dtype=complex)
             else:
                 m.python_matrix = FractalMatrix7.identity()
             return m
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

        # Python Fallback
        if not self.ptr:
            if self.backend == "numpy":
                 for _ in range(steps):
                    v = np.sin(self.python_matrix @ v)
                    # STABILITY FIX: Normalize
                    norm = np.linalg.norm(v)
                    if norm > 1e-12:
                        v = v / norm
                 return v
            else:
                 # Structural Backend (Pure Python)
                 # Input 'initial_state' might be list or numpy array. Convert to FractalVector7
                 fv = FractalVector7(initial_state)
                 for _ in range(steps):
                     # v = sin(M @ v)
                     fv = self.python_matrix.dot(fv).apply_complex_sin().normalize()
                 # Return list or numpy array? Keep consistent with input if possible, or list
                 return np.array(fv.v) if 'numpy' in sys.modules else fv.v
            
        # We need mutable buffers for the C function to write back into
        # But wait, C function arg is 'vec_real', 'vec_imag'. 
        # Is it in/out? Yes, logic says load, loop, save back.
        
        v_real = np.ascontiguousarray(v.real, dtype=np.float64)
        v_imag = np.ascontiguousarray(v.imag, dtype=np.float64)
        
        p_real = v_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_imag = v_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        self.lib.Matrix7x7_Evolve(self.ptr, p_real, p_imag, ctypes.c_int(steps))
        
        return v_real + 1j * v_imag

    def evolve_batch(self, batch_states, steps=1):
        """
        Simulate Superposition of N Fractal Nodes.
        Input: (N, 7) array of complex numbers.
        Output: (N, 7) array of evolved states.
        """
        batch = np.array(batch_states, dtype=complex)
        if batch.ndim != 2 or batch.shape[1] != 7:
            raise ValueError("Batch must be (N, 7)")
            
        N = batch.shape[0]

        # Python Fallback
        if not self.ptr:
            # Vectorized implementation
            # v (N,7), M (7,7). v' = sin( (M @ v.T).T ) = sin( v @ M.T )
            M_T = self.python_matrix.T
            v = batch
            for _ in range(steps):
                v = np.sin(v @ M_T)
                # STABILITY FIX: Batch Normalize
                # Norm along axis 1 (rows)
                norms = np.linalg.norm(v, axis=1, keepdims=True)
                # Avoid div by zero
                norms[norms < 1e-12] = 1.0
                v = v / norms
            return v
        
        # Contiguous buffers representing the flat batch
        v_real = np.ascontiguousarray(batch.real.flatten(), dtype=np.float64)
        v_imag = np.ascontiguousarray(batch.imag.flatten(), dtype=np.float64)
        
        p_real = v_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_imag = v_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Call C++ Engine
        self.lib.Matrix7x7_BatchEvolve(self.ptr, p_real, p_imag, ctypes.c_int(N), ctypes.c_int(steps))
        
        # Reconstruct
        res_real = v_real.reshape((N, 7))
        res_imag = v_imag.reshape((N, 7))
        
        return res_real + 1j * res_imag
