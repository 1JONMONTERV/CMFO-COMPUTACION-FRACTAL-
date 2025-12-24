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
        Analisis Graph to find all unique input symbols.
        """
        self.code = ""
        self.indent_level = 0
        
        # 1. Collect Symbols
        symbols = self._collect_symbols(op)
        raw_syms = list(set(s.name for s in symbols))
        
        # HACK: Enforce 'v', 'h' order for compatibility with fixed C++ Bridge
        # The C++ bridge calls (v_ptr, h_ptr, out_ptr).
        # So we MUST generate kernel(const float* v, const float* h, ...)
        uniq_symbols = []
        if 'v' in raw_syms: uniq_symbols.append('v')
        if 'h' in raw_syms: uniq_symbols.append('h')
        
        # Add any other symbols (dynamic mode) - Future proofing
        for s in sorted(raw_syms):
            if s not in uniq_symbols:
                uniq_symbols.append(s)
        
        # 2. Generate Signature
        # extern "C" __global__ void kernel(const float* sym1, const float* sym2, ..., float* out, int N)
        args = [f"const float* {s}" for s in uniq_symbols]
        args.append("float* out")
        args.append("int N")
        signature = ", ".join(args)
        
        self.emit(f'extern "C" __global__ void {kernel_name}({signature}) {{')
        self.indent()
        
        # 3. Thread Indexing
        self.emit("int idx = threadIdx.x + blockIdx.x * blockDim.x;")
        self.emit("if (idx >= N) return;")
        self.emit("")
        
        # 4. Load Constants
        self.emit("const float PHI = 1.6180339887f;")
        self.emit("")
        
        # 5. Dynamic Unrolled Loads
        self.emit("// 7D Unrolled Loads")
        for sym in uniq_symbols:
            for i in range(7):
                self.emit(f"float {sym}{i} = {sym}[idx*7 + {i}];")
        self.emit("")
        
        # 6. Generate Logic
        self.emit("// Fused Logic")
        for i in range(7):
            result_expr = self.visit(op, subscript=i)
            self.emit(f"out[idx*7 + {i}] = {result_expr};")
            
        self.dedent()
        self.emit("}")
        return self.code

    def _collect_symbols(self, node):
        syms = []
        if isinstance(node, Symbol):
            return [node]
        elif isinstance(node, AlgebraicOp):
            return self._collect_symbols(node.left) + self._collect_symbols(node.right)
        elif isinstance(node, GeometricOp):
            return self._collect_symbols(node.input_node)
        return []

    def visit(self, node: FractalNode, subscript: int) -> str:
        """
        Recursive visitor that returns a C++ expression string.
        """
        if not isinstance(node, FractalNode):
             # Debugging type issues
             print(f"DEBUG: Visit called with {type(node)} -> {node}")
             
        if isinstance(node, Symbol):
            # Maps symbol 'v' to local var 'v0', 'v1' etc
            return f"{node.name}{subscript}"
            
        elif isinstance(node, Constant):
            return f"{node.value}f"
            
        if isinstance(node, AlgebraicOp) or type(node).__name__ == 'AlgebraicOp':
            l = self.visit(node.left, subscript)
            r = self.visit(node.right, subscript)
            
            if node.op_type in ['add', '+']:
                return f"({l} + {r})"
            elif node.op_type in ['sub', '-']:
                return f"({l} - {r})"
            elif node.op_type in ['mul', '*']:
                return f"({l} * {r})"
            elif node.op_type in ['div', '/']:
                return f"({l} / {r})"
            elif node.op_type == 'pow':
                # Use CUDA intrinsic
                return f"powf({l}, {r})"
            elif node.op_type == 'min':
                return f"fminf({l}, {r})" # CUDA min
            else:
                raise ValueError(f"Unknown AlgebraicOp type: {node.op_type} in {node}")
                
        elif isinstance(node, GeometricOp) or type(node).__name__ == 'GeometricOp':
            inp = self.visit(node.input_node, subscript)
            if node.op_type == 'sqrt':
                return f"sqrtf({inp})"
            elif node.op_type == 'step':
                return f"({inp} >= 0.0f ? 1.0f : -1.0f)" # Sign function
            elif node.op_type == 'gamma':
                return f"tgammaf({inp})"
            elif node.op_type == 'sin':
                return f"sinf({inp})"
             
        raise ValueError(f"Unknown Node: {node}")

    def emit(self, line):
        self.code += "  " * self.indent_level + line + "\n"
        
    def indent(self):
        self.indent_level += 1
        
    def dedent(self):
        self.indent_level -= 1
