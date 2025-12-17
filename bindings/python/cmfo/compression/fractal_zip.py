
import os
import lzma
import shutil
import pickle
import math
import sys

class FractalCompressor:
    """
    CMFO Fractal Zip.
    1. Compresses data using LZMA2 (High Ratio).
    2. Stores the result in 'Hyper-Memory' (Disk-based Fractal Swap).
    """
    def __init__(self, swap_dir="fractal_archive"):
        self.swap_dir = swap_dir
        if not os.path.exists(swap_dir):
            os.makedirs(swap_dir)
        print(f"[INIT] Fractal Compressor linked to Hyper-Memory: {swap_dir}")

    def compress_file(self, file_path):
        """
        Compresses a file and sends it directly to Hyper-Memory.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"[ZIP] Reading {file_path}...")
        with open(file_path, 'rb') as f:
            data = f.read()
        
        orig_size = len(data)
        
        # 1. Compress
        print("[ZIP] Applying Fractal Compression (LZMA2)...")
        compressed_data = lzma.compress(data, preset=9)
        comp_size = len(compressed_data)
        if orig_size > 0:
            ratio = (1 - (comp_size / orig_size)) * 100
        else:
            ratio = 0
        
        # 2. Store in Hyper-Memory
        fname = os.path.basename(file_path)
        key = f"{fname}.fractal"
        
        # Use Phi-based hashing
        phi = 1.6180339887
        shard_id = int((hash(fname) * phi) % 10) 
        shard_dir = os.path.join(self.swap_dir, f"shard_{shard_id}")
        if not os.path.exists(shard_dir):
            os.makedirs(shard_dir)
            
        store_path = os.path.join(shard_dir, key)
        
        print(f"[MEM] Offloading to Hyper-Memory Shard {shard_id}...")
        with open(store_path, 'wb') as f:
            pickle.dump(compressed_data, f)
            
        print(f"[SUCCESS] Stored '{key}' at {store_path}")
        print(f"   Original: {orig_size/1024:.2f} KB")
        print(f"   Fractal:  {comp_size/1024:.2f} KB")
        print(f"   Ratio:    {ratio:.2f}% Saved")
        
        return store_path

    def decompress_file(self, archive_path, output_path):
        """
        Retrieves from Hyper-Memory and Decompresses.
        """
        if not os.path.exists(archive_path):
             raise FileNotFoundError(f"Archive not found: {archive_path}")
             
        print(f"[MEM] Retrieving from Hyper-Memory: {archive_path}")
        with open(archive_path, 'rb') as f:
            compressed_data = pickle.load(f)
            
        print("[UNZIP] Expanding Fractal Data...")
        data = lzma.decompress(compressed_data)
        
        with open(output_path, 'wb') as f:
            f.write(data)
            
        print(f"[SUCCESS] Restored to {output_path}")
