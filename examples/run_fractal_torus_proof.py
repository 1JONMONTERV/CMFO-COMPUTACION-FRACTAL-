
import sys
import os
import numpy as np

# Add the bindings path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, '..', 'bindings', 'python')
sys.path.append(bindings_path)

try:
    from cmfo.topology.fractal_torus import FractalTorus
except ImportError as e:
    print(f"Error importing CMFO: {e}")
    sys.exit(1)

def save_ppm(grid, filename):
    """
    Save the binary grid as a Portable Pixel Map (PPM/PBM) image.
    Format P1 (ASCII Bitmap) is simple and widely supported.
    """
    height, width = grid.shape
    with open(filename, 'w') as f:
        f.write("P1\n")
        f.write(f"{width} {height}\n")
        # Go row by row
        for row in grid:
            # Join 0s and 1s with spaces or just cat them
            line = ' '.join(map(str, row))
            f.write(line + "\n")
    print(f"Saved visualization to: {filename}")

def run_proof():
    print("==================================================")
    print("   CMFO FRACTAL TORUS - GEOMETRIC PROOF v1.0")
    print("   Target: 1024x1024 Binary Mesh (Mara Binaria)")
    print("   Topology: Torus (Periodic)")
    print("   Operator: 7x7 Phi-Weighted Matrix")
    print("==================================================")

    # Initialize
    ft = FractalTorus(size=1024, kernel_size=7)
    
    print("\n[Phase 1] Evolution Dynamics (Searching for Attractors)")
    print("-" * 50)
    
    final_metrics = {}
    for i in range(1, 51):
        ft.step()
        metrics = ft.measure_geometry()
        
        if i % 10 == 0:
            print(f"Step {i:03d} | Tensor Trace: {metrics['tensor_trace_mean']:.5f} | "
                  f"Entropy: {metrics['entropy']:.5f} | "
                  f"Angle: {metrics['mean_angle']:.5f} rad")
        final_metrics = metrics

    print("\n[Phase 2] Exact Geometric Measurement Results")
    print("-" * 50)
    print(f"Final Tensor Trace (Avg Local Density): {final_metrics['tensor_trace_mean']:.9f}")
    print(f"Geometric Entropy (Complexity):         {final_metrics['entropy']:.9f}")
    print(f"Mean Angular Phase (Gradient):          {final_metrics['mean_angle']:.9f} rad")
    print(f"Attractor State Reached:                {final_metrics['in_attractor']}")
    
    print("\n[Phase 3] Generating Visual Proof")
    
    output_path = os.path.join(current_dir, 'fractal_torus_state.pbm')
    
    # Try Matplotlib first, else fallback to PPM
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(ft.grid, cmap='binary', interpolation='nearest')
        plt.title(f"Mara Binaria 1024x1024 - Fractal Torus (Step 50)\nEntropy: {final_metrics['entropy']:.4f}")
        plt.axis('off')
        png_path = output_path.replace('.pbm', '.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {png_path}")
    except ImportError:
        print("Matplotlib not found. Using native PBM (Portable Bitmap) writer.")
        save_ppm(ft.grid, output_path)

    print("PROOF COMPLETE.")

if __name__ == "__main__":
    run_proof()
