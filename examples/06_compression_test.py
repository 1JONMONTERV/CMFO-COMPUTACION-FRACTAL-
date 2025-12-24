"""
06_compression_test.py
----------------------
Demonstrates the experimental CMFO Semantic Compression.
Verifies that data survives the round-trip (lossless).
"""

import cmfo.compression
import zlib

def main():
    print("--- CMFO Compression Benchmark ---")
    
    # Text with repetitive semantic structure but varied characters
    text = "The quick brown fox jumps over the lazy dog. " * 100
    original_size = len(text.encode('utf-8'))
    
    print(f"Original Size: {original_size} bytes")
    
    # Compress
    compressed = cmfo.compression.compress_text(text)
    cmfo_size = len(compressed)
    
    print(f"CMFO Archive Size: {cmfo_size} bytes (includes 56-byte semantic header)")
    print(f"Ratio: {original_size / cmfo_size:.2f}x")
    
    # Validation 1: Header extraction (instant semantic match)
    header = cmfo.compression.get_semantic_header(compressed)
    print(f"Semantic Signature (First Dim): {header[0]:.4f}")
    
    # Validation 2: Lossless Round-trip
    restored = cmfo.compression.decompress_text(compressed)
    
    if restored == text:
        print(">> SUCCESS: Data restored perfectly.")
    else:
        print(">> FAILURE: Data corruption detected.")

if __name__ == "__main__":
    main()
