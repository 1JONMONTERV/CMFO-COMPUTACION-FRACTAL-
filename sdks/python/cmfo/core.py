"""
CMFO Python SDK - Core Bindings
================================
Pure Python wrapper around CMFO C ABI using ctypes.
No external dependencies except CMFO core library.
"""

import ctypes
import os
from typing import Optional, List, Tuple
from pathlib import Path

# Locate CMFO library
def _find_library():
    """Find CMFO shared library"""
    lib_names = {
        'win32': 'cmfo.dll',
        'darwin': 'libcmfo.dylib',
        'linux': 'libcmfo.so'
    }
    
    import sys
    lib_name = lib_names.get(sys.platform, 'libcmfo.so')
    
    # Search paths
    search_paths = [
        Path(__file__).parent / 'lib',
        Path.cwd() / 'build' / 'lib',
        Path('/usr/local/lib'),
        Path('/usr/lib'),
    ]
    
    for path in search_paths:
        lib_path = path / lib_name
        if lib_path.exists():
            return str(lib_path)
    
    # Try system search
    return lib_name

# Load library
_lib = ctypes.CDLL(_find_library())

# Types
class CMFOVersion(ctypes.Structure):
    _fields_ = [
        ('major', ctypes.c_uint16),
        ('minor', ctypes.c_uint16),
        ('patch', ctypes.c_uint16)
    ]

class CMFOVec7(ctypes.Structure):
    _fields_ = [('v', ctypes.c_double * 7)]
    
    def to_list(self) -> List[float]:
        return list(self.v)
    
    @classmethod
    def from_list(cls, values: List[float]):
        if len(values) != 7:
            raise ValueError("Vector must have exactly 7 components")
        vec = cls()
        for i, v in enumerate(values):
            vec.v[i] = v
        return vec

class CMFOConfig(ctypes.Structure):
    _fields_ = [
        ('mode', ctypes.c_uint32),
        ('memory_limit_bytes', ctypes.c_uint64),
        ('license_key', ctypes.c_char_p),
        ('audit_log_path', ctypes.c_char_p),
        ('flags', ctypes.c_uint32)
    ]

# Constants
CMFO_OK = 0
CMFO_MODE_STUDY = 0x01
CMFO_MODE_RESEARCH = 0x02
CMFO_MODE_ENTERPRISE = 0x04

# Function signatures
_lib.cmfo_get_version.restype = CMFOVersion
_lib.cmfo_init.argtypes = [ctypes.POINTER(CMFOConfig)]
_lib.cmfo_init.restype = ctypes.c_void_p
_lib.cmfo_destroy.argtypes = [ctypes.c_void_p]
_lib.cmfo_get_error.argtypes = [ctypes.c_void_p]
_lib.cmfo_get_error.restype = ctypes.c_char_p

_lib.cmfo_parse.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(CMFOVec7)]
_lib.cmfo_parse.restype = ctypes.c_int

_lib.cmfo_solve.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
_lib.cmfo_solve.restype = ctypes.c_int

_lib.cmfo_compose.argtypes = [ctypes.c_void_p, ctypes.POINTER(CMFOVec7), ctypes.POINTER(CMFOVec7), ctypes.POINTER(CMFOVec7)]
_lib.cmfo_compose.restype = ctypes.c_int

_lib.cmfo_distance.argtypes = [ctypes.c_void_p, ctypes.POINTER(CMFOVec7), ctypes.POINTER(CMFOVec7), ctypes.POINTER(ctypes.c_double)]
_lib.cmfo_distance.restype = ctypes.c_int

# High-level API
class CMFO:
    """CMFO Context Manager"""
    
    def __init__(self, mode=CMFO_MODE_STUDY, license_key=None):
        config = CMFOConfig()
        config.mode = mode
        config.memory_limit_bytes = 0
        config.license_key = license_key.encode() if license_key else None
        config.audit_log_path = None
        config.flags = 0
        
        self._ctx = _lib.cmfo_init(ctypes.byref(config))
        if not self._ctx:
            raise RuntimeError("Failed to initialize CMFO")
    
    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            _lib.cmfo_destroy(self._ctx)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.__del__()
    
    def parse(self, text: str) -> List[float]:
        """Parse text to 7D vector"""
        vec = CMFOVec7()
        result = _lib.cmfo_parse(self._ctx, text.encode(), ctypes.byref(vec))
        if result != CMFO_OK:
            error = _lib.cmfo_get_error(self._ctx)
            raise RuntimeError(f"Parse error: {error.decode()}")
        return vec.to_list()
    
    def solve(self, equation: str) -> str:
        """Solve equation"""
        solution_ptr = ctypes.c_char_p()
        result = _lib.cmfo_solve(self._ctx, equation.encode(), ctypes.byref(solution_ptr))
        if result != CMFO_OK:
            error = _lib.cmfo_get_error(self._ctx)
            raise RuntimeError(f"Solve error: {error.decode()}")
        solution = solution_ptr.value.decode()
        ctypes.c_free(solution_ptr)
        return solution
    
    def compose(self, v: List[float], w: List[float]) -> List[float]:
        """Compose two vectors"""
        vec_v = CMFOVec7.from_list(v)
        vec_w = CMFOVec7.from_list(w)
        result_vec = CMFOVec7()
        
        result = _lib.cmfo_compose(self._ctx, ctypes.byref(vec_v), ctypes.byref(vec_w), ctypes.byref(result_vec))
        if result != CMFO_OK:
            error = _lib.cmfo_get_error(self._ctx)
            raise RuntimeError(f"Compose error: {error.decode()}")
        
        return result_vec.to_list()
    
    def distance(self, v: List[float], w: List[float]) -> float:
        """Calculate fractal distance"""
        vec_v = CMFOVec7.from_list(v)
        vec_w = CMFOVec7.from_list(w)
        dist = ctypes.c_double()
        
        result = _lib.cmfo_distance(self._ctx, ctypes.byref(vec_v), ctypes.byref(vec_w), ctypes.byref(dist))
        if result != CMFO_OK:
            error = _lib.cmfo_get_error(self._ctx)
            raise RuntimeError(f"Distance error: {error.decode()}")
        
        return dist.value

def get_version() -> Tuple[int, int, int]:
    """Get CMFO version"""
    ver = _lib.cmfo_get_version()
    return (ver.major, ver.minor, ver.patch)

# Example usage
if __name__ == "__main__":
    print(f"CMFO Python SDK v{'.'.join(map(str, get_version()))}")
    
    with CMFO() as cmfo:
        # Parse
        vec = cmfo.parse("verdad")
        print(f"verdad = {vec}")
        
        # Solve
        solution = cmfo.solve("2x + 3 = 7")
        print(f"Solution:\n{solution}")
        
        # Compose
        v1 = [1, 0, 0, 0, 0, 0, 0]
        v2 = [0, 1, 0, 0, 0, 0, 0]
        composed = cmfo.compose(v1, v2)
        print(f"Composed: {composed}")
        
        # Distance
        d = cmfo.distance(v1, v2)
        print(f"Distance: {d}")
