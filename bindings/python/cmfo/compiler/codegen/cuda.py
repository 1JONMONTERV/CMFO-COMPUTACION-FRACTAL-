"""
CMFO CUDA Generator
===================
Translates Fractal IR into optimized CUDA C++ code.
"The Sniper" implementation: Generates unrolled 7D kernels.
"""

from cmfo.compiler.ir import *

class CUDAGenerator:
    def __init__(self):
        self.code = ""
        self.indent_level = 0

    def generate_kernel(self, op: FractalNode, kernel_name="fused_kernel"):
        """
        Generates a complete CUDA kernel string for a given operation graph.
        """
        self.code = ""
        self.indent_level = 0
        
        # Header
        self.emit(f'extern "C" __global__ void {kernel_name}(const float* v, const float* h, float* out, int N) {{')
        self.indent()
        
        # Thread Indexing
        self.emit("int idx = threadIdx.x + blockIdx.x * blockDim.x;")
        self.emit("if (idx >= N) return;")
        self.emit("")
        
        # Load Constants (Snippet)
        self.emit("// Constants")
        self.emit("const float PHI = 1.6180339887f;")
        self.emit("")
        
        # Semantic Unrolling (The Sniper)
        self.emit("// 7D Unrolled Loads")
        for i in range(7):
            self.emit(f"float v{i} = v[idx*7 + {i}];")
            self.emit(f"float h{i} = h[idx*7 + {i}];")
        self.emit("")
        
        # Generate Logic
        self.emit("// Fused Logic")
        # We recursively generate the expression for each dimension 0..6
        # For this prototype, we assume the graph is uniform across dimensions (element-wise)
        # We generate the code for a *scalar* operation and apply it to v{i}, h{i}
        
        for i in range(7):
            result_expr = self.visit(op, subscript=i)
            self.emit(f"out[idx*7 + {i}] = {result_expr};")
            
        self.dedent()
        self.emit("}")
        return self.code

    def visit(self, node: FractalNode, subscript: int) -> str:
        """
        Recursive visitor that returns a C++ expression string.
        """
        if isinstance(node, Symbol):
            # Maps symbol 'v' to local var 'v0', 'v1' etc
            return f"{node.name}{subscript}"
            
        elif isinstance(node, Constant):
            return f"{node.value}f"
            
        elif isinstance(node, AlgebraicOp):
            l = self.visit(node.left, subscript)
            r = self.visit(node.right, subscript)
            
            if node.op_type == 'add':
                return f"({l} + {r})"
            elif node.op_type == 'mul':
                return f"({l} * {r})"
            elif node.op_type == 'div':
                return f"({l} / {r})"
            elif node.op_type == 'pow':
                # Use CUDA intrinsic
                return f"powf({l}, {r})"
                
        elif isinstance(node, GeometricOp):
             # Harder, not implemented in this minimal prototype
             # Gamma ops mix dimensions, so they break the "element-wise" simple unroll
             # They require a Matrix Mul unroll
             return "0.0f /* TODO GammaOp */"
             
        return "0.0f"

    def emit(self, line):
        self.code += "  " * self.indent_level + line + "\n"
        
    def indent(self):
        self.indent_level += 1
        
    def dedent(self):
        self.indent_level -= 1
