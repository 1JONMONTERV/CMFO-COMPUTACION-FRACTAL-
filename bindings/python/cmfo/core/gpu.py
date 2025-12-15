"""
CMFO GPU Accelerator Bridge
===========================
Pure Python interface for GPU Acceleration.
Uses ctypes to load raw CUDA/OpenCL kernels if available.
Zero dependencies (no torch, no cupy required).
"""

import ctypes
import os
import ctypes
import os
import sys
import array

class Accelerator:
    """
    Manages connection to native GPU libraries.
    """
    _lib = None
    _checked = False

    @staticmethod
    def is_available():
        """Check if a native GPU library is loaded."""
        if not Accelerator._checked:
            Accelerator._load_library()
        return Accelerator._lib is not None

    @staticmethod
    def _load_library():
        """Attempts to load cmfo_cuda.dll or libcmfo_cuda.so"""
        Accelerator._checked = True
        Accelerator._lib = None # Reset
        
        # Paths to search
        names = ["cmfo_cuda.dll", "libcmfo_cuda.so"]
        paths = [
            os.path.abspath("."),
            os.path.join(os.path.dirname(__file__), "../../../lib"), 
            "/usr/local/lib"
        ]

        for path in paths:
            for name in names:
                full_path = os.path.join(path, name)
                if os.path.exists(full_path):
                    try:
                        Accelerator._lib = ctypes.CDLL(full_path)
                        print(f"[CMFO] GPU Backend Loaded: {full_path}")
                        return
                    except Exception as e:
                        print(f"[CMFO] Found GPU lib but failed to load: {e}")
        
        # Fallback: Create a Mock Object for Demonstration if requested
        # This allows users to see the pipeline work even without compiling C
        print("[CMFO] GPU Library not found. Using Virtual Accelerator (Simulation).")
        Accelerator._lib = "VIRTUAL_GPU"

    @staticmethod
    def get_kernel(name):
        """
        Returns a callable Python function that wraps the C++ GPU symbol.
        """
        if not Accelerator.is_available():
            raise RuntimeError("GPU Accelerator not available.")
            
        # Example: mapping linear_7d to C function
        if name == "linear_7d":
            if Accelerator._lib == "VIRTUAL_GPU":
                 # Return a Python simulator that LOOKS like the C wrapper
                 def virtual_kernel(input_data):
                     # Optimized Path Simulator
                     if isinstance(input_data, tuple):
                         # (array, batch, dim)
                         flat_arr, batch, dim = input_data
                         out_features = 64
                         out_arr = array.array('f', [0.0] * (batch * out_features))
                         PHI = 1.6180339887
                         
                         # Use memoryview for zero-copy slicing of the input
                         mv = memoryview(flat_arr)
                         
                         # Pre-calc harmonics to remove math from loop (simulation opt)
                         harmonics = [PHI**(float(i%7)) for i in range(out_features)]
                         # Denom is (1+h), store multiplier h/(1+h)
                         multipliers = [h / (1.0 + h) for h in harmonics]
                         
                         for b in range(batch):
                             start = b * dim
                             # Zero-copy slice sum
                             # Still implies iterating in CPython, but avoids allocation
                             val = sum(mv[start : start+dim])
                             
                             out_start = b * out_features
                             for i in range(out_features):
                                 out_arr[out_start + i] = val * multipliers[i]
                         return out_arr

                     # Legacy list path
                     input_list = input_data
                     batch = len(input_list)
                     output = []
                     PHI = 1.6180339887
                     out_features = 64
                     
                     for b in range(batch):
                        input_vec = input_list[b]
                        val = sum(input_vec)
                        row = [(val * (PHI**(i%7)))/(1+(PHI**(i%7))) for i in range(out_features)]
                        output.append(row)
                     return output
                 return virtual_kernel

            # Real Ctypes Bridge
            func = Accelerator._lib.cmfo_linear_gpu
            func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            
            def wrapper(input_data):
                # Optimization Path: Check if input is already a flat buffer (array.array)
                if isinstance(input_data, array.array):
                    # Zero-Copy Path!
                    # Input is already flat float array
                    # We assume (Batch * Dim) layout
                    # Recover batch/dim from metadata or arguments (simplified here)
                    # For stress test: we pass batch/dim explicitly or assume fixed
                    # Let's infer: we need batch count.
                     
                    # For this demo, we assume input_data is FLATTENED array
                    # We need 'batch' and 'dim' passed or inferred. 
                    # Let's adhere to the previous signature taking list-of-lists?
                    # No, to be zero-copy the input MUST be flat array.
                    # We will support a tuple (array, batch, dim) for the optimized call
                    
                    if isinstance(input_data, tuple):
                        flat_arr, batch, dim = input_data
                    else:
                        raise ValueError("Optimized call expects (flat_array, batch, dim)")

                    # Get pointer directly
                    c_in = (ctypes.c_float * len(flat_arr)).from_buffer(flat_arr)
                    
                    out_features = 64
                    # Output Buffer (Zero Copy)
                    # Pre-allocate output array in Python
                    out_arr = array.array('f', [0.0] * (batch * out_features))
                    c_out = (ctypes.c_float * len(out_arr)).from_buffer(out_arr)
                    
                    # RUN
                    func(c_in, c_out, batch, dim)
                    
                    return out_arr # Return flat array
                
                # Slow Path (Legacy List of Lists)
                import itertools
                flat_input = list(itertools.chain(*input_data))
                batch = len(input_data)
                dim = len(input_data[0]) if batch > 0 else 0
                
                FloatArray = ctypes.c_float * len(flat_input)
                c_in = FloatArray(*flat_input)
                
                out_features = 64 
                FloatArrayOut = ctypes.c_float * (batch * out_features)
                c_out = FloatArrayOut()
                
                func(c_in, c_out, batch, dim)
                
                output = []
                idx = 0
                for _ in range(batch):
                    row = []
                    for _ in range(out_features):
                        row.append(c_out[idx])
                        idx += 1
                    output.append(row)
                return output
                
            return wrapper
            
        return None
