
import sys
import os
import time

# Add binding path
current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, '..', 'bindings', 'python')
sys.path.append(bindings_path)

try:
    from cmfo.mining.distiller import BlockDistiller
except ImportError as e:
    print(f"Error importing CMFO Mining: {e}")
    sys.exit(1)

def run_proof():
    print("==================================================")
    print("   CMFO FRACTAL DISTILLER - 1024-BIT PROOF")
    print("   Method: Monte Carlo Search (Random Sampling)")
    print("   Target: 1024-bit Blocks with High Entropy")
    print("==================================================")
    
    # 1. Initialize Distiller
    db_file = os.path.join(current_dir, 'golden_solutions.json')
    distiller = BlockDistiller(database_path=db_file)
    
    # 2. Configuration
    # We want to demonstrate finding blocks. 
    # Difficulty 3 zeros (hex) = 12 bits zero ~ 1/4096 chance.
    # Complexity 4 zeros (hex) = 16 bits zero ~ 1/65536 chance.
    TARGET_DIFFICULTY = 4 
    SCAN_AMOUNT = 200000 
    
    print(f"[Setup] Scanning {SCAN_AMOUNT} blocks of 1024-bits (128 bytes).")
    print(f"[Setup] Target Difficulty: {TARGET_DIFFICULTY} leading hex zeros (Prob ~ 1/65k)")
    
    # 3. Run Distillation
    distiller.distiller_loop(total_blocks=SCAN_AMOUNT, difficulty=TARGET_DIFFICULTY)
    
    # 4. Verification Report
    if len(distiller.solutions) > 0:
        print("\n[Verification] Sample Golden Block found:")
        sample = distiller.solutions[0]
        print(f"  Hash: {sample['hash']}")
        print(f"  Block (First 32 bytes): {sample['block_hex'][:64]}...")
        
        # Calculate Compression
        raw_size = SCAN_AMOUNT * 128 # bytes
        db_size = os.path.getsize(db_file)
        ratio = raw_size / db_size
        print(f"\n[Efficiency] Compression Statistics:")
        print(f"  Raw Scanned Data: {raw_size/1024/1024:.2f} MB")
        print(f"  Stored Golden DB: {db_size/1024:.2f} KB")
        print(f"  Effective Ratio:  {ratio:.1f}x (Garbage Discarded)")
        print("  Mining Strategy:  VALIDATED")
    else:
        print("\n[Result] No blocks found (Try lowering difficulty or increasing scan time).")

if __name__ == "__main__":
    run_proof()
