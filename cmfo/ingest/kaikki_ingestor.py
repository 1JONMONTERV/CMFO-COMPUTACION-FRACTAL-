"""
CMFO D9: Kaikki Ingestor
========================
Massive Ingestion of Kaikki.org Spanish Dictionary.

Pipeline:
Raw JSONL -> Extract Glosses -> SemanticCompiler -> D9 Storage

Input: D:/CMFO_DATA/ontology/raw_source/kaikki_es.jsonl
Target: D:/CMFO_DATA/shards/ & /vectors/
"""

import json
import io
import sys
import time
from pathlib import Path

# Internal dependencies
try:
    from .compiler import SemanticCompiler
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compiler import SemanticCompiler


SOURCE_FILE = Path("D:/CMFO_DATA/ontology/raw_source/kaikki_es.jsonl")

def ingest_kaikki(limit: int = 0):
    """
    Ingest Kaikki Dictionary.
    limit: Max items to process (0 = all)
    """
    if not SOURCE_FILE.exists():
        print(f"Error: Source file not found at {SOURCE_FILE}")
        return
        
    print(f"Starting Ingestion from {SOURCE_FILE}")
    print("Initializing Compiler...")
    compiler = SemanticCompiler()
    
    count = 0
    success_count = 0
    start_time = time.time()
    
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if limit > 0 and count >= limit:
                break
                
            count += 1
            if count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = count / elapsed
                params = f"Processed: {count} | Success: {success_count} | Rate: {rate:.1f} items/s"
                sys.stdout.write(f"\r{params}")
                sys.stdout.flush()
                
            try:
                entry = json.loads(line)
                
                # Extract Term
                term = entry.get("word", "")
                if not term:
                    continue
                    
                # Extract Senses (Definitions)
                senses = entry.get("senses", [])
                
                for sense in senses:
                    glosses = sense.get("glosses", [])
                    if not glosses:
                        continue
                        
                    definition = glosses[0] # Take first gloss
                    
                    # Compile
                    # We pass 'kaikki' as source
                    compiler.compile(term, definition, source="kaikki_v1")
                    success_count += 1
                    
            except Exception as e:
                # Log errors but don't stop industrial process
                # print(f"\nError processing line {count}: {e}")
                pass
                
    total_time = time.time() - start_time
    print(f"\n\nINGESTION COMPLETE.")
    print(f"Total Items: {count}")
    print(f"Compiled Vectors: {success_count}")
    print(f"Time: {total_time:.2f}s")


if __name__ == "__main__":
    # By default process all, or pass a limit arg
    limit = 0
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except:
            pass
            
    # For initial test, let's limit to 10,000 to prove flow without waiting hours
    # User can run full batch later
    if limit == 0:
        print("Ingesting ALL items (Massive Industrial Run)...")
    else:
        print(f"Ingesting first {limit} items...")
        
    ingest_kaikki(limit=limit)
