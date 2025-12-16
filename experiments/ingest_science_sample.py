"""
CMFO D13: Sample Ingestion Verification
=======================================
Runs the ScientificExtractor on the downloaded Math dataset.
Demonstrates the extraction of Definitions and Relations.
"""

from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

# Force UTF-8 for Windows Console
sys.stdout.reconfigure(encoding='utf-8')

from cmfo.ingest.science.extractor import ScientificExtractor
from cmfo.cognition.axiom_mapper import AxiomMapper

TARGET_FILE = Path("D:/CMFO_DATA/science/raw/math/math-00-part-1.zip")

def run_sample():
    print(f"Reading Scientific Sample: {TARGET_FILE}")
    
    if not TARGET_FILE.exists():
        print("File not found! (Download failed?)")
        return
        
    extractor = ScientificExtractor(TARGET_FILE)
    mapper = AxiomMapper() # Bridge is None for this test
    
    count = 0
    stats = {"definitions": 0, "relations": 0, "theorems": 0, "proofs": 0}
    
    for paper in extractor.stream_papers():
        layers = extractor.extract_layers(paper["text"])
        
        # Only print interesting papers
        if layers["definitions"] or layers["relations"]:
            count += 1
            print(f"\n--- Paper ID: {paper['id']} ---")
            
            for d in layers["definitions"][:2]: # Show max 2
                print(f"[DEF] {d[:100]}...")
                # Try mapping (Experimental)
                vec = mapper.formal_to_vector(d)
                if vec:
                    print(f"      -> Mapped to Axioms: {vec[:3]}...")
                    
            for r in layers["relations"][:2]:
                print(f"[REL] {r[:100]}...")
            
            for t in layers.get("theorems", [])[:2]:
                print(f"[THM] {t[:100]}...")
                
            for p in layers.get("proofs", [])[:1]:
                print(f"[PRF] {p[:150]}...")
            
            stats["definitions"] += len(layers["definitions"])
            stats["relations"] += len(layers["relations"])
            stats["theorems"] = stats.get("theorems", 0) + len(layers.get("theorems", []))
            stats["proofs"] = stats.get("proofs", 0) + len(layers.get("proofs", []))
            
            if count >= 5: # Sample size
                break
                
    print(f"\nSummary of First {count} Valid Papers:")
    print(f"Definitions Extracted: {stats['definitions']}")
    print(f"Relations Extracted:   {stats['relations']}")
    print(f"Theorems Extracted:    {stats['theorems']}")
    print(f"Proofs Extracted:      {stats['proofs']}")

if __name__ == "__main__":
    run_sample()
