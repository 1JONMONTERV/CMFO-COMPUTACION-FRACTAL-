"""
CMFO Fractal JIT Manager
========================
Handles initialization of the Native JIT Bridge (NVRTC).
Compiles Fractal IR -> CUDA -> PTX -> Execution.
"""

import ctypes
import os
import hashlib
from typing import List
from .codegen import CUDAGenerator
from .ir import FractalNode

class FractalJIT:
    _lib = None
    _checked = False
    _kernel_cache = {} # Map code_hash -> kernel_id

    @staticmethod
    def is_available():
        if not FractalJIT._checked:
            FractalJIT._load_library()
        return FractalJIT._lib is not None

    @staticmethod
    def _load_library():
        FractalJIT._checked = True
        
        # 0. Setup Dependencies (Windows specific)
        if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
            cuda_path = os.environ.get('CUDA_PATH')
            
            paths_to_add = []
            if cuda_path:
                paths_to_add.append(os.path.join(cuda_path, 'bin'))
            
            # Known location fallback (Hardcoded based on exploration)
            known_cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
            if os.path.exists(known_cuda):
                paths_to_add.append(known_cuda)

            for p in paths_to_add:
                try:
                    os.add_dll_directory(p)
                    print(f"[CMFO JIT] Added DLL Directory: {p}")
                except Exception as e:
                    print(f"[CMFO JIT] Failed to add {p}: {e}")

        # Paths to search for cmfo_jit.dll
        paths = [
            os.path.abspath("lib"),
            os.path.abspath("bin"),
            os.path.join(os.path.dirname(__file__), "../../../lib"),
            "." # Current dir
        ]
        
        dll_name = "cmfo_jit.dll" if os.name == 'nt' else "libcmfo_jit.so"
        # print(f"DEBUG: Searching for {dll_name}...")
        
        for path in paths:
            full_path = os.path.join(path, dll_name)
            # print(f"DEBUG: Checking {full_path}")
            if os.path.exists(full_path):
                # print(f"DEBUG: Found at {full_path}")
                try:
                    FractalJIT._lib = ctypes.CDLL(full_path)
                    
                    # Bind Init Function
                    FractalJIT._lib.cmfo_jit_init.argtypes = []
                    FractalJIT._lib.cmfo_jit_init.restype = ctypes.c_int
                    
                    # Initialize CUDA Driver
                    res = FractalJIT._lib.cmfo_jit_init()
                    if res != 0:
                        print("[CMFO JIT] Warning: cuInit failed (No GPU?). Native mode disabled.")
                        FractalJIT._lib = None
                        return

                    # --- BIND NEW STATEFUL API ---
                    
                    # int cmfo_jit_load_cache(src, name)
                    FractalJIT._lib.cmfo_jit_load_cache.argtypes = [
                        ctypes.c_char_p, 
                        ctypes.c_char_p
                    ]
                    FractalJIT._lib.cmfo_jit_load_cache.restype = ctypes.c_int

                    # int cmfo_jit_launch_cache(id, v, h, out, N)
                    FractalJIT._lib.cmfo_jit_launch_cache.argtypes = [
                        ctypes.c_int,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_int
                    ]
                    FractalJIT._lib.cmfo_jit_launch_cache.restype = ctypes.c_int
                    
                    print(f"[CMFO JIT] Loaded Native Backend: {full_path} (Stateful)")
                    return
                except Exception as e:
                    print(f"[CMFO JIT] Failed to load found DLL: {e}")

        print("[CMFO JIT] Native bridge (cmfo_jit.dll) not found or failed to load.")
        print("[CMFO JIT] 'The Sniper' is running in Python Simulation Mode (Slow).")

    @staticmethod
    def compile_and_run(ir_graph: FractalNode, v: List[float], h: List[float]) -> List[float]:
        """
        Main Entry Point for JIT Execution.
        """
        if len(v) != len(h):
            raise ValueError("Input dimensions mismatch")
        
        # 1. Generate Source (Fingerprinting)
        gen = CUDAGenerator()
        source_code = gen.generate_kernel(ir_graph, kernel_name="cmfo_autogen")
        
        # 2. Native Path
        if FractalJIT.is_available():
            # CACHING LOGIC
            src_bytes = source_code.encode('utf-8')
            code_hash = hashlib.md5(src_bytes).hexdigest()
            
            kernel_id = -1
            if code_hash in FractalJIT._kernel_cache:
                kernel_id = FractalJIT._kernel_cache[code_hash]
                # print(f"[JIT] Cache Hit: Kernel {kernel_id}")
            else:
                name_bytes = b"cmfo_autogen"
                kernel_id = FractalJIT._lib.cmfo_jit_load_cache(src_bytes, name_bytes)
                if kernel_id < 0:
                     raise RuntimeError("JIT Compilation Failed Native Side.")
                FractalJIT._kernel_cache[code_hash] = kernel_id
                # print(f"[JIT] Cache Miss: Compiled new Kernel {kernel_id}")

            # Prepare Memory (This part is still per-run overhead, avoidable in v4.0 with resident tensors)
            import array
            import itertools
            
            if isinstance(v[0], list):
                 v_flat = array.array('f', itertools.chain(*v))
                 h_flat = array.array('f', itertools.chain(*h))
                 N = len(v)
            else:
                 v_flat = array.array('f', v)
                 h_flat = array.array('f', h)
                 N = len(v_flat) // 7

            out_flat = array.array('f', [0.0] * (N * 7))

            # Pointers
            v_ptr, _ = v_flat.buffer_info()
            h_ptr, _ = h_flat.buffer_info()
            out_ptr, _ = out_flat.buffer_info()

            # 3. Launch
            ret = FractalJIT._lib.cmfo_jit_launch_cache(
                ctypes.c_int(kernel_id),
                ctypes.c_void_p(v_ptr),
                ctypes.c_void_p(h_ptr),
                ctypes.c_void_p(out_ptr),
                ctypes.c_int(N)
            )
            
            if ret != 0:
                raise RuntimeError("Native JIT Execution Failed!")

            # Reconstruct output
            output = []
            for i in range(N):
                output.append(out_flat[i*7 : (i+1)*7].tolist())
            return output

        else:
            # Fallback: Python Simulation
            print("WARNING: Native JIT not available. Falling back to Mock execution.")
            return [[0.0]*7] * (len(v) if isinstance(v[0], list) else len(v)//7)
