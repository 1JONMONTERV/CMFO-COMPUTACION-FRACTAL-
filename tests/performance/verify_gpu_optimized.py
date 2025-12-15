
import time
import sys
import os
import array

# Add bindings from local
sys.path.insert(0, os.path.abspath("bindings/python"))

from cmfo.layers.linear import CMFOLinear
from cmfo.core.gpu import Accelerator

def stress_test_optimized():
    print("==================================================")
    print("   [+] CMFO BRIDGE OPTIMIZED BENCHMARK [+]")
    print("==================================================")
    
    # 1. Setup Layer
    linear = CMFOLinear(in_features=1024, out_features=64)
    linear.to("cuda")
    
    if not Accelerator.is_available():
        print("    [X] Bridge Failed.")
        return

    # 2. Generate Massive Load (Flat Buffer)
    BATCH_SIZE = 5000 
    DIM = 1024
    print(f"\n[2] Generating FLAT Load: {BATCH_SIZE} vectors of dim {DIM}...")
    
    # Create one massive flat array
    total_floats = BATCH_SIZE * DIM
    # Fast init
    flat_data = array.array('f', [0.1] * total_floats)
    
    print(f"    [!] Payload Size: {flat_data.buffer_info()[1] * 4 / 1024 / 1024:.2f} MB")
    
    # 3. Execution Loop
    print("\n[3] Executing Kernel via ZERO-COPY Bridge...")
    
    kernel = Accelerator.get_kernel("linear_7d")
    
    t0 = time.time()
    
    # Pass Tuple (buffer, batch, dim) to trigger optimized path
    output_buffer = kernel((flat_data, BATCH_SIZE, DIM))
    
    t1 = time.time()
    delta = t1 - t0
    
    print(f"    [!] Time: {delta:.4f} seconds")
    print(f"    [!] Throughput: {BATCH_SIZE / delta:,.2f} vectors/sec")
    
    # 4. Correctness Check
    print("\n[4] Verification...")
    # output_buffer is a flat array ('f')
    print(f"    Output Type: {type(output_buffer)}")
    print(f"    Length: {len(output_buffer)} (Expected {BATCH_SIZE*64})")
    
    # Check first value
    # Energy of [0.1]*1024 = 102.4
    # Harmonic 0 = 1.0
    # Res = 102.4 * 1 / 2 = 51.2
    val = output_buffer[0]
    print(f"    First Val: {val:.4f} (Expected ~51.2)")
    
    print("\n==================================================")
    print("   [+] OPTIMIZED RUN COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    stress_test_optimized()
