"""
CMFO D13: Full Science Processor (Parallel)
===========================================
Massive Extraction of Algebraic Structures from arXiv PDFs.
input:  D:/CMFO_DATA/science/raw/math/math-00-part-1.zip
output: D:/CMFO_DATA/shards/science/math_01.jsonl
"""

import json
import time
import sys
import os
import zipfile
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

# Adjust path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from cmfo.ingest.science.extractor import ScientificExtractor

CONFIG = {
    "source": Path("D:/CMFO_DATA/science/raw/math/math-00-part-1.zip"),
    "output_dir": Path("D:/CMFO_DATA/shards/science"),
    "batch_size": 50
}

def process_chunk(filenames: List[str]) -> List[Dict]:
    """Worker: Opens zip locally, processes list of PDF filenames"""
    results = []
    
    # Each worker opens the zip independently to avoid lock contention/pickling
    try:
        # Re-instantiate extractor logic locally or use helper
        # We need pypdf here 
        import io
        from pypdf import PdfReader
        
        # We assume the Extractor class logic is reusable but we need the file handle fn
        # Let's simplify and do raw pypdf + extractor.extract_layers logic here
        # or use the class if it supports single file processing (it streams).
        
        # Let's use the class methodology but tailored for single generic "pdf bytes"
        
        with zipfile.ZipFile(CONFIG["source"], 'r') as zf:
            for name in filenames:
                try:
                    with zf.open(name) as f:
                        pdf_bytes = io.BytesIO(f.read())
                        reader = PdfReader(pdf_bytes)
                        text = ""
                        for page in reader.pages:
                            t = page.extract_text()
                            if t: text += t + "\n"
                            
                    # Extraction Logic (Heuristic)
                    # We instantiate extractor just for its regex methods
                    # Passing None as path since we use our own text
                    extractor = ScientificExtractor(Path(".")) 
                    layers = extractor.extract_layers(text)
                    
                    if layers["definitions"] or layers["relations"]:
                        results.append({
                            "id": name,
                            "layers": layers
                        })
                        
                except Exception as e:
                    # Corrupt PDF or Parse Error
                    continue
                    
    except Exception as e:
        print(f"Worker Error: {e}")
        
    return results

def main():
    print(f"CMFO Science Processor (Multicore)")
    print(f"==================================")
    print(f"Source: {CONFIG['source']}")
    
    if not CONFIG['source'].exists():
        print("Source file missing.")
        return
        
    # Create Output Dir
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    out_file = CONFIG["output_dir"] / "math_01.jsonl"
    
    # 1. List Files (Main Process)
    print("Indexing Zip contents...")
    with zipfile.ZipFile(CONFIG["source"], 'r') as zf:
        all_files = [f for f in zf.namelist() if f.endswith('.pdf')]
        
    print(f"Found {len(all_files)} PDFs.")
    
    # 2. Dispatch
    start_time = time.time()
    total_extracted = 0
    total_defs = 0
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        chunk = []
        
        for fname in all_files:
            chunk.append(fname)
            if len(chunk) >= CONFIG["batch_size"]:
                futures.append(executor.submit(process_chunk, list(chunk)))
                chunk = []
        
        if chunk:
            futures.append(executor.submit(process_chunk, chunk))
            
        print(f"Dispatched {len(futures)} tasks. Processing...")
        
        # 3. Collect & Write
        with open(out_file, "w", encoding="utf-8") as f_out:
            completed = 0
            for future in as_completed(futures):
                res = future.result()
                
                for item in res:
                    f_out.write(json.dumps(item) + "\n")
                    total_extracted += 1
                    total_defs += len(item["layers"]["definitions"])
                
                completed += 1
                
                # Progress
                if completed % 5 == 0:
                    elapsed = time.time() - start_time
                    percent = (completed / len(futures)) * 100
                    rate = (completed * CONFIG["batch_size"]) / elapsed
                    sys.stdout.write(f"\rProgress: {percent:.1f}% | Extract: {total_extracted} | Defs: {total_defs} | PDF/s: {rate:.1f}")
                    sys.stdout.flush()
                    
    print(f"\n\nPROCESSING COMPLETE.")
    print(f"Papers Mined: {total_extracted}")
    print(f"Definitions:  {total_defs}")
    print(f"Output:       {out_file}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
