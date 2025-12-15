
import sys
sys.path.insert(0, 'bindings/python')
from cmfo.core.structural import FractalVector7
from cmfo.compiler.jit import FractalJIT

def auto_jit_demo():
    print("="*40)
    print("  CMFO PHASE 4: AUTO-JIT DEMO")
    print("="*40)

    # 1. Symbolic Mode
    print("[1] Creating Symbolic Vectors...")
    v = FractalVector7.symbolic('v')
    h = FractalVector7.symbolic('h')
    
    print(f"    v: {v}")
    print(f"    h: {h}")
    
    # 2. Lazy Operations
    print("\n[2] Building Graph (Lazy Add/Mul)...")
    # This looks like math, but expects graph building
    result = v * 0.5 + h
    
    print(f"    Expression: {result}")
    
    if result.is_lazy:
        print("    ✅ Result is Lazy (Graph Captured)")
        print(f"    Graph Node: {result._node}")
    else:
        print("    ❌ Failed: Result is Eager")

    # 3. Compilation Trigger (Preview)
    print("\n[3] JIT Trigger (Simulation)...")
    try:
        # In full implementation, we would do: result.compute(inputs=...)
        # Here we manually verify we CAN generate code from this object
        from cmfo.compiler.codegen.cuda import CUDAGenerator
        gen = CUDAGenerator()
        code = gen.generate_kernel(result._node, "auto_jit_kernel")
        print("    ✅ Code Generation Successful!")
        print("    Snippet:")
        print('\n'.join(code.split('\n')[:8]) + "...")
    except Exception as e:
        print(f"    ❌ CodeGen Failed: {e}")

if __name__ == "__main__":
    auto_jit_demo()
