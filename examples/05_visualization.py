"""
05_visualization.py
-------------------
Demonstrates the Visualizer and Text Bridge capabilities.
"""

import cmfo.bridge
import cmfo.vis

def main():
    text = "CMFO is Geometric"
    
    print(f"Analyzing: '{text}'")
    
    # 1. Convert to Tensor
    tensor = cmfo.bridge.text_to_tensor(text)
    
    # 2. Visualize
    cmfo.vis.plot_tensor_ascii(tensor, label="Semantic Geometry")
    
    # 3. Trajectory
    seq = cmfo.bridge.encode_sequence(text)
    cmfo.vis.plot_attractor_trajectory(seq)

if __name__ == "__main__":
    main()
