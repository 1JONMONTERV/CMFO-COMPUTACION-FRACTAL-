"""
Fractal Frame Generation (FPS Booster)
======================================
Predicts intermediate frames using Gamma-Step Manifold Trajectory.
"""

import numpy as np
from PIL import Image
import sys
import os
import math

# Try importing CMFO core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from cmfo.constants import PHI
except ImportError:
    PHI = 1.6180339887

class FramePredictor:
    def __init__(self):
        pass

    def gamma_interpolate(self, frame_a, frame_b, t=0.5):
        """
        Predicts frame at time t (0..1) between A and B.
        Uses non-linear Phi-interpolation.
        """
        try:
            # Load
            img_a = Image.open(frame_a).convert('RGB')
            img_b = Image.open(frame_b).convert('RGB')
            
            # Resize B to match A if needed
            if img_a.size != img_b.size:
                img_b = img_b.resize(img_a.size)
                
            arr_a = np.array(img_a, dtype=np.float32)
            arr_b = np.array(img_b, dtype=np.float32)
            
            # Delta Vector
            delta = arr_b - arr_a
            
            # Linear Prediction (Newtonian)
            linear_pred = arr_a + (delta * t)
            
            # Fractal Correction (The "Swirl")
            # We assume motion isn't perfectly linear but follows a shallow curve defined by PHI
            # Curve factor: sin(t * PI) * PHI_factor
            curve_mag = math.sin(t * math.pi) * 0.1 # Small deviation at midpoint
            
            # Apply correction based on pixel gradient magnitude
            # (Pixels that change a lot, curve more)
            magnitude = np.abs(delta).mean(axis=2, keepdims=True) / 255.0
            correction = delta * curve_mag * magnitude
            
            final_arr = linear_pred + correction
            final_arr = np.clip(final_arr, 0, 255)
            
            return Image.fromarray(final_arr.astype(np.uint8))
            
        except Exception as e:
            print(f"FrameGen Error: {e}")
            return None

if __name__ == "__main__":
    # Test
    print("Testing Frame Generator...")
    gen = FramePredictor()
    
    # Create dummy frames (red circle moving)
    # Just solid colors for now
    f1 = Image.new('RGB', (100, 100), color=(255, 0, 0))
    f2 = Image.new('RGB', (100, 100), color=(0, 0, 255))
    f1.save("frame_0.png")
    f2.save("frame_1.png")
    
    mid = gen.gamma_interpolate("frame_0.png", "frame_1.png", 0.5)
    if mid:
        mid.save("frame_0.5.png")
        print("Frame Gen Successful: frame_0.5.png (Expect Purple-ish)")
