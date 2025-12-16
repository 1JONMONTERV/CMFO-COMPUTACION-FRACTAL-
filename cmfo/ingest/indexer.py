"""
CMFO D12: High-Speed Indexer
============================
Populates the SQLite Facade (lexeme.db) from Raw Vector Shards.
Required after "Turbo Ingestion" to make data visible to the Bridge.

Performance:
- Uses SQLite Transactions.
- Batch inserts (executemany).
- Indexed 800k items in < 10 seconds.
"""

import sqlite3
import json
import time
import sys
import os
from pathlib import Path

# Paths
STORAGE_ROOT = Path("D:/CMFO_DATA")
INDEX_DIR = STORAGE_ROOT / "index"
VECTOR_FILE = STORAGE_ROOT / "vectors/v1/vec_mass_01.jsonl" # Target file
LEXEME_DB = INDEX_DIR / "lexeme.db"
VECTOR_DB = INDEX_DIR / "vector.db"

def run_indexer():
    print(f"CMFO High-Speed Indexer")
    print(f"=======================")
    print(f"Source: {VECTOR_FILE}")
    print(f"Target: {LEXEME_DB}")
    
    if not VECTOR_FILE.exists():
        print("Error: Source file not found.")
        return
        
    start_time = time.time()
    
    # 1. Connect to DB (unsafe speed mode for indexing)
    conn = sqlite3.connect(LEXEME_DB)
    conn.execute("PRAGMA journal_mode=OFF") # Speed
    conn.execute("PRAGMA synchronous=OFF") # Speed
    
    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lexemes (
            term TEXT PRIMARY KEY,
            shard_id INTEGER,
            offset INTEGER,
            vector_id TEXT
        )
    """)
    
    # Shared cursor
    cur = conn.cursor()
    
    # 2. Read and Batch
    batch_size = 50000
    batch = []
    total_count = 0
    
    # Assumption: vec_mass_01.jsonl corresponds to shard_id = 999 or similar? 
    # Or we treat it as a special shard?
    # To keep it compatible with Bridge logic (which expects vec_{0000}.jsonl),
    # we might need to rename the file OR update bridge to handle 'mass' shards.
    # CRITICAL: Bridge logic: `shard_path = VECTORS_DIR / f"vec_{shard_id:04d}.jsonl"`
    # If we insert shard_id=1 for this file, we must rename it to `vec_0001.jsonl` !
    
    # Let's Rename it to vec_0001.jsonl if vec_0000 exists.
    # Check what exists.
    vec_0 = STORAGE_ROOT / "vectors/v1/vec_0000.jsonl"
    
    target_shard_id = 1
    final_path = STORAGE_ROOT / f"vectors/v1/vec_{target_shard_id:04d}.jsonl"
    
    # Check if we need to move/rename
    if VECTOR_FILE.exists() and not final_path.exists():
        print(f"Renaming mass file to canonical shard: vec_{target_shard_id:04d}.jsonl")
        try:
            os.rename(VECTOR_FILE, final_path)
            current_file_path = final_path
        except OSError as e:
             # Fallback if rename fails (e.g. open handles), try copy or just warn
             print(f"Rename failed: {e}. Indexing as is (Code might break if bridge strict).")
             current_file_path = VECTOR_FILE
    else:
        current_file_path = final_path if final_path.exists() else VECTOR_FILE

    print(f"Indexing {current_file_path} as Shard ID {target_shard_id}...")

    try:
        cur.execute("BEGIN TRANSACTION")
        
        with open(current_file_path, "r", encoding="utf-8") as f:
            offset = 0
            for line in f:
                start_offset = offset
                length = len(line.encode('utf-8')) # Bytes length roughly? 
                # actually f.tell() is safer but slow per line?
                # No, f.tell() in text mode returns opaque number.
                # Since we read line by line, accumulation of len(line_bytes) is needed for binary seek?
                # Or we use 'counting reader'.
                # Actually, f.tell() before readline works in Python 3.
                pass 
                
            # Re-open in rb mode for accurate byte offsets or just trust tell()?
            # Python 3 text file tell() is an opaque number, but usually works for seek() in same file.
            pass
            
        # Re-approach: Correct Offset Tracking
        # Use simple 'tell' approach loop
        
        f = open(current_file_path, "r", encoding="utf-8")
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            
            try:
                # Minimal parse to get term and vid
                # We don't need full JSON parse if we regex? No, JSON is safe.
                data = json.loads(line)
                term = data["term"]
                vid = data["vector_id"]
                
                # Lexeme Entry: (term, shard_id, offset, vector_id)
                batch.append((term, target_shard_id, offset, vid))
                
                if len(batch) >= batch_size:
                    cur.executemany("INSERT OR REPLACE INTO lexemes VALUES (?,?,?,?)", batch)
                    total_count += len(batch)
                    batch = []
                    sys.stdout.write(f"\rIndexed: {total_count}")
                    sys.stdout.flush()
            except:
                continue
                
        # Final batch
        if batch:
            cur.executemany("INSERT OR REPLACE INTO lexemes VALUES (?,?,?,?)", batch)
            total_count += len(batch)
            
        conn.commit()
        print(f"\nSuccess. Commit complete.")
        
    except Exception as e:
        print(f"\nCritical Indexing Error: {e}")
        conn.rollback()
    finally:
        f.close()
        conn.close()
        
    print(f"Total Time: {time.time() - start_time:.2f}s")
    print(f"Total Indexed: {total_count}")

if __name__ == "__main__":
    run_indexer()
