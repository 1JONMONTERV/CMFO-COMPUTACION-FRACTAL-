import ctypes
import os
import sys
import glob

# Types
c_double = ctypes.c_double
Vector7 = c_double * 7
Matrix7 = (c_double * 7) * 7

class CMFOCore:
    def __init__(self):
        self.lib = self._load_library()
        if not self.lib:
            raise RuntimeError("Could not load cmfo_core library")
        
        self._setup_bindings()
        
    def _load_library(self):
        # Search paths
        paths = [
            "./build/libcmfo_core.so",
            "./build/cmfo_core.dll",
            "./build/Release/cmfo_core.dll",
            "../build/libcmfo_core.so",
        ]
        # Also try glob for system-specific names
        paths.extend(glob.glob("./build/**/*cmfo_core*", recursive=True))
        
        for p in paths:
            if os.path.exists(p) and os.path.isfile(p):
                try:
                    return ctypes.CDLL(p)
                except:
                    continue
        
        print("Warning: libcmfo_core not found. Bindings will fail.")
        return None

    def _setup_bindings(self):
        # cmfo_phi
        self.lib.cmfo_phi.restype = c_double
        
        # cmfo_tensor7(double out[7], const double a[7], const double b[7])
        self.lib.cmfo_tensor7.argtypes = [Vector7, Vector7, Vector7]
        self.lib.cmfo_tensor7.restype = None

        # cmfo_mat7_inv(double out[7][7], const double M[7][7])
        self.lib.cmfo_mat7_inv.argtypes = [Matrix7, Matrix7]
        self.lib.cmfo_mat7_inv.restype = ctypes.c_int

    def phi(self):
        return self.lib.cmfo_phi()
        
    def tensor7(self, a, b):
        out = Vector7()
        c_a = Vector7(*a)
        c_b = Vector7(*b)
        self.lib.cmfo_tensor7(out, c_a, c_b)
        return list(out)

    def mat7_inv(self, M):
        out = Matrix7()
        c_M = Matrix7()
        for i in range(7):
            for j in range(7):
                c_M[i][j] = M[i][j]
                
        res = self.lib.cmfo_mat7_inv(out, c_M)
        if res == 0:
            raise ValueError("Singular Matrix")
            
        py_out = []
        for i in range(7):
            row = []
            for j in range(7):
                row.append(out[i][j])
            py_out.append(row)
        return py_out

def demo():
    try:
        core = CMFOCore()
        print(f"Loaded CMFO Core. Phi = {core.phi()}")
        
        a = [1.0]*7
        b = [2.0]*7
        t = core.tensor7(a, b)
        print(f"Tensor Product ([1...], [2...]): {t}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Ensure you have built the shared library (cmake -DBUILD_SHARED=ON).")

if __name__ == "__main__":
    demo()
