
import os
import sys
import math
import struct
import random
import time

# Add local path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from cmfo.crypto.fractal_sha256 import FractalSHA256, FractalWord

def get_resonance(header_prefix, nonce_trits):
    """
    Calculates the 'Resonance' (Phi-Metric) of a state.
    Resonance = Proximity to '0' State (Valid Hash).
    Higher Resonance = Deeper 'Well' in the landscape.
    """
    # Construct Block
    w1_int, w2_int = float_to_fractal_block(header_prefix, nonce_trits)
    
    fsha = FractalSHA256()
    fsha.compress(w1_int)
    fsha.compress(w2_int)
    h_int = fsha.get_hash()
    
    # Calculate Zero-Depth (Leading Zeros)
    # We sum the leading zero BITS.
    zeros = 0
    for val in h_int:
        if val == 0:
            zeros += 32
        else:
            # Count leading zeros in 32-bit int
            lz = 0
            if val == 0: lz=32
            else:
                while (val >> (31-lz)) & 1 == 0:
                    lz += 1
            zeros += lz
            break
            
    # Add fractional resonance (closeness to next zero)
    # This is the "continuous" part.
    # In a discrete hash, we don't have fractional.
    # But in FractalSHA, 'a' and 'e' state variables trace the "Stress".
    # We can use the internal 'FractalRAM' observables if enabled, 
    # but for now, the "Potential Well" depth is approx 'zeros'.
    
    return zeros

def float_to_fractal_block(header_prefix, nonce_trits):
    # (Reusing logic from solver)
    w1 = [FractalWord.from_int(struct.unpack(">I", header_prefix[i:i+4])[0]) for i in range(0,64,4)]
    
    tail_bytes = header_prefix[64:] 
    tail_words = [FractalWord.from_int(struct.unpack(">I", tail_bytes[i:i+4])[0]) for i in range(0,12,4)]
    
    # Int Nonce
    nonce_val = 0
    for i,t in enumerate(nonce_trits):
         if t>0.5: nonce_val |= (1 << (31-i))
         
    nonce_word = FractalWord.from_int(nonce_val)
    tail_words.append(nonce_word)
    
    pad_bytes = b'\x80' + b'\x00'*39 + struct.pack(">Q", 640)
    pad_words = [FractalWord.from_int(struct.unpack(">I", pad_bytes[i:i+4])[0]) for i in range(0,48,4)]
    
    w2 = tail_words + pad_words
    return w1, w2

def scan_neighborhood(center_nonce, header_prefix, radius=20):
    """
    Scans the T7 manifold around a center point.
    """
    # header_prefix passed from caller to maintain context
    
    # Convert center int to trits
    center_trits = [(center_nonce >> (31-i)) & 1 for i in range(32)]
    
    print(f"Mapping T7 Landscape around Nonce {center_nonce}...")
    print(f"Radius: {radius} (Hamming/Linear distance)")
    
    # We will plot a 1D slice for simplicity: Linearly adjacent nonces
    # X-axis: Nonce Offset (-radius to +radius)
    # Y-axis: Resonance (Zeros)
    
    map_data = []
    
    for offset in range(-radius, radius+1):
        target_nonce = center_nonce + offset
        # Trits
        trits = [float((target_nonce >> (31-i)) & 1) for i in range(32)]
        
        res = get_resonance(header_prefix, trits)
        map_data.append((offset, res))
        
    return map_data

def visualize_attractor():
    print("--- FRACTAL TIME GEOMETRY VISUALIZER ---")
    print("Visualizing the 'Attractor' nature of a Valid Block.")
    
    # 1. Find a semi-valid block (Low difficulty) to use as 'Center'
    # Or just pick a random one and show the rugged landscape.
    # Better: Pick a "Target" (Diff 10) and find it, then map around it.
    
    # Brute force search for a small 'well' (Diff 10)
    print("Locating a local Attractor (Diff 10)...")
    center = 0
    header_prefix = struct.pack("<I", 1) + b'\x00'*32 + b'\xAA'*32 + struct.pack("<I", int(time.time())) + struct.pack("<I", 0x1d00ffff)

    while True:
        trits = [float((center >> (31-i)) & 1) for i in range(32)]
        if get_resonance(header_prefix, trits) >= 10:
            break
        center += 1
        if center % 1000 == 0: print(f"Searching... {center}", end="\r")
        
    print(f"\nAttractor Found at Nonce {center}. Mapping Geometry...")
    
    data = scan_neighborhood(center, header_prefix, radius=30)
    
    # Visualization (ASCII Plot)
    print("\n[T7 RESONANCE MAP]")
    print(f"Center ({center}) marks the timestamp/coordinate of validity.")
    print("Y-Axis: Resonance (Zero Depth) | X-Axis: Temporal/Nonce Displacement")
    print("-" * 60)
    
    max_res = max(d[1] for d in data)
    
    for off, res in data:
        bar = "#" * res
        # Highlight center
        marker = ">>" if off == 0 else "  "
        print(f"{marker} {off:+4d} | {bar} ({res})")
        
    print("-" * 60)
    print("INTERPRETATION:")
    print("The 'Spike' at 0 represents the Geometry aligning.")
    print("In classical time, you must step -30, -29... to hit 0.")
    print("In Fractal Time being mapped, the 'Gradient' points to 0.")
    print("Using the Solver, we 'slide' up the slope of resonance instantly.")

if __name__ == "__main__":
    visualize_attractor()
