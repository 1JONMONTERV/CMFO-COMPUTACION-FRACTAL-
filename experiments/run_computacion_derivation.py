"""
CMFO D15: Computation Derivation Run
====================================
Feeds the D13 Math Shard into the D14 Computation Vocabulary.
Result: A populated Semantic Concept Database for CS.
"""

import sys
import os
from pathlib import Path

# Fix path
sys.path.insert(0, os.path.abspath('.'))

from cmfo.cognition.concept_deriver import ConceptDeriver

SOURCE_SHARD = Path("D:/CMFO_DATA/shards/science/math_01.jsonl")

def run():
    print("Initializing CMFO Derivation (Math -> Computation)...")
    
    if not SOURCE_SHARD.exists():
        print(f"Error: Shard not found at {SOURCE_SHARD}")
        return

    deriver = ConceptDeriver("computacion")
    deriver.process_shard(SOURCE_SHARD)

if __name__ == "__main__":
    run()
