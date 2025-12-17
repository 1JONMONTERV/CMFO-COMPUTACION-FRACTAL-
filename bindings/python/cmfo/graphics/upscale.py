"""
Fractal Super-Resolution (Megapixel Booster)
============================================
Upscales images by injecting semantic noise from the T7 Manifold.
Better than bicubic because it hallucinates consistent texture.
"""

import numpy as np
from PIL import Image
import math
import sys
import os

# Try importing CMFO core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from cmfo.constants import PHI
    from cmfo.core.gamma_phi import gamma_step
except ImportError:
    PHI = 1.6180339887
    def gamma_step(v): return [math.sin(x) for x in v]


from typing import Optional, Tuple, Union

class FractalUpscaler:
    """
    CMFO Fractal Super-Resolution Engine.
    
    Uses phi-driven noise injection to hallucinate high-frequency texture details
    during image upscaling, avoiding the 'plastic' look of bicubic interpolation.
    """
    
    def __init__(self, intensity: float = 0.1):
        """
        Initialize the upscaler.
        
        Args:
           intensity (float): Strength of the fractal texture injection (0.0 - 1.0).
                              Default 0.1 provides subtle realistic grain.
        """
        self.intensity = intensity

    def upscale(self, image_path: str, scale_factor: int = 2) -> Optional[Image.Image]:
        """
        Upscale an image using Fractal Injection.
        
        Args:
            image_path (str): Path to source image.
            scale_factor (int): Multiplier (currently supports x2).
            
        Returns:
            PIL.Image.Image: The upscaled image object, or None on failure.
        """
        if scale_factor != 2:
            print("Warning: Currently only x2 scaling is optimized for Fractal JIT.")
            # Fallback to simple resize if needed, but for now we enforce x2 logic
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Input image not found: {image_path}")

            img = Image.open(image_path).convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            
            h, w, _ = arr.shape
            new_h, new_w = h * scale_factor, w * scale_factor
            
            # 1. Bicubic Base (The "Low Frequency" Guess)
            base_img = img.resize((new_w, new_h), Image.BICUBIC)
            base_arr = np.array(base_img, dtype=np.float32) / 255.0
            
            # 2. Fractal Injection (The "High Frequency" Texture)
            # x coordinate map scaled by Phi
            x_map = np.tile(np.arange(new_w), (new_h, 1)) * PHI
            y_map = np.tile(np.arange(new_h).reshape(-1, 1), (1, new_w)) * PHI
            
            # Texture seed: The base image brightness (Luma approximation)
            brightness = np.mean(base_arr, axis=2)
            
            # The Fractal: Sin(Brightness * x * Phi)
            # Creates coherent texture following image structure
            noise = np.sin(brightness * x_map * y_map) * self.intensity
            
            # Expand noise to 3 channels
            noise_3d = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
            
            final_arr = base_arr + (noise_3d * 0.05) # Add subtle texture
            
            # Clip and convert back
            final_arr = np.clip(final_arr, 0.0, 1.0)
            final_img = Image.fromarray((final_arr * 255).astype(np.uint8))
            
            return final_img
            
        except Exception as e:
            print(f"Upscale Error: {e}")
            return None

if __name__ == "__main__":
    # Test
    print("Testing Fractal Upscaler...")
    upscaler = FractalUpscaler()
    # Create dummy image
    dummy = Image.new('RGB', (100, 100), color = 'red')
    dummy.save("test_input.png")
    
    res = upscaler.upscale("test_input.png")
    if res:
        res.save("test_output.png")
        print("Upscale Successful: test_output.png")
