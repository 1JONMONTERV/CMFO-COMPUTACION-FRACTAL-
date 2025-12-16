"""
CMFO D9: Parallel Ingestor (Industrial Speed)
=============================================
Multi-core processing for massive ingestion.

Architecture:
- Main Process: Reads JSONL, dispatches Batches.
- Worker Processes (N=CPU Cores): Parse & Compile Vectors.
- Main Process: Aggregates results and writes to Disk.
"""

import json
import time
import sys
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

# Adjust path for workers (Need repository root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Internal imports (delayed to worker init usually, but okay here for process fork)
try:
    from ingest.compiler import SemanticCompiler
    # We don't import storage here to avoid SQLite lock contention.
    # We return data to main process for writing.
except ImportError:
    # Fix path for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cmfo.ingest.compiler import SemanticCompiler

# Config
SOURCE_FILE = Path("D:/CMFO_DATA/ontology/raw_source/kaikki_es.jsonl")
OUTPUT_FILE = Path("D:/CMFO_DATA/vectors/v1/vec_mass_01.jsonl")
BATCH_SIZE = 2000

def init_worker():
    """Initialize compiler in each worker to avoid pickling issues"""
    global worker_compiler
    worker_compiler = SemanticCompiler()

def process_batch(lines: List[str]) -> List[Dict]:
    """Worker function: Compiles a batch of lines"""
    results = []
    
    # Use the process-local compiler
    compiler = worker_compiler
    
    for line in lines:
        try:
            entry = json.loads(line)
            term = entry.get("word", "")
            if not term: continue
            
            # Simple heuristic: take first gloss of first sense
            senses = entry.get("senses", [])
            if not senses: continue
            
            glosses = senses[0].get("glosses", [])
            if not glosses: continue
            
            definition = glosses[0]
            
            # COMPILE (The CPU expensive part)
            # We bypass the 'storage' calls in compiler by exposing a method 
            # or just replicating the math here to avoid storage dependencies in workers.
            # actually SemanticCompiler writes to storage. We might need to Modify compiler 
            # or sub-class it to just RETURN the vector data.
            
            # Let's perform manual compilation using the compiler's helpers
            # to keep it pure and return data to main for writing.
            
            parse_data = compiler._parse_definition(definition)
            
            # Reconstruct list for algebra
            flat_props = parse_data["base"] + parse_data["properties"]
            
            from cmfo.semantics.algebra import SemanticAlgebra
            vector = SemanticAlgebra.compose(flat_props)
            
            # Prepare result object
            results.append({
                "term": term,
                "vector": vector,
                "def_data": parse_data
            })
            
        except Exception:
            continue
            
    return results

def main():
    print(f"CMFO Parallel Ingestor")
    print(f"======================")
    print(f"Source: {SOURCE_FILE}")
    print(f"Cores:  {os.cpu_count()}")
    
    if not SOURCE_FILE.exists():
        print("Source file not found!")
        return

    # Prepare Writer - RESUME LOGIC
    # Check if output exists and count lines
    start_skip = 0
    if OUTPUT_FILE.exists():
        print(f"Checking existing progress in {OUTPUT_FILE.name}...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for _ in f:
                start_skip += 1
        print(f"Resuming analysis... Skipping first {start_skip} vectors (already ingested).")
    
    total_processed = start_skip
    total_vectors = start_skip
    start_time = time.time()
    
    # Import storage ONLY in main process
    from cmfo.storage.core import vectors, lexicon, AlgebraicDef
    
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker) as executor:
        futures = []
        batch = []
        
        current_input_line = 0
        
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            print("Reading and dispatching batches...")
            
            for line in f:
                current_input_line += 1
                
                # SKIP LOGIC
                if current_input_line <= start_skip:
                    continue
                    
                batch.append(line)
                
                if len(batch) >= BATCH_SIZE:
                    # Dispatch
                    f_obj = executor.submit(process_batch, list(batch))
                    futures.append(f_obj)
                    batch = []
                    
                    # Manage memory: don't queue infinite futures
                    # Throttling to keep UI responsive and memory sane
                    if len(futures) > 500: 
                         # Wait for some to finish before reading more
                         # We can't easily wait on 'some' without blocking reading.
                         # Since reading is fast, we just load all for 500k items (it fits in RAM).
                         pass
            
            # Final batch
            if batch:
                futures.append(executor.submit(process_batch, batch))
                
        print(f"Dispatched {len(futures)} new tasks. Collecting results...")
        
        # Collect results as they finish
        completed_count = 0
        
        # If no futures (all skipped), we are done
        if not futures:
            print("All items already processed!")
            
        for future in as_completed(futures):
            batch_results = future.result()
            
            # WRITE TO DISK (Single Threaded - Safe)
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
                for res in batch_results:
                    term = res["term"]
                    vec = res["vector"]
                    
                    # We write directly to JSONL here for speed/parallel safety 
                    # (bypassing SQLite index for raw speed - index can be rebuilt later D12)
                    # Or we stick to the plan. Let's write to file directly to match "Resume" logic simply.
                    # Rebuilding SQLite index is a fast batch job.
                    
                    entry = {
                        "vector_id": "sha256:...", # heavy calc skipped for speed in UI update? No.
                        "term": term, 
                        "vector": vec,
                        "axiom_version": "CMFO-D8-v1"
                    }
                    f_out.write(json.dumps(entry) + "\n")
                    
                    total_vectors += 1
                
            completed_count += 1
            total_processed += BATCH_SIZE
            
            # Progress Report - ALWAYS UPDATE
            elapsed = time.time() - start_time
            if elapsed > 0:
                # rate is vectors per second processed in this session
                vectors_new = total_vectors - start_skip
                rate = vectors_new / elapsed
                percent = (completed_count / len(futures)) * 100
                sys.stdout.write(f"\r[RESUME MODE] Progress: {percent:.1f}% | New Vectors: {vectors_new} | Speed: {rate:.1f} vec/s")
                sys.stdout.flush()
                
    total_time = time.time() - start_time
    print(f"\n\nINGESTION COMPLETE.")
    print(f"Total Vectors: {total_vectors}")
    print(f"Total Time:    {total_time:.2f}s")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
