"""
CMFO D15: Concept Deriver
=========================
The Mining Engine.
Reads raw scientific layers, filters by Domain Vocabulary,
calculates Algebraic Vectors, and stores valid Concepts.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict

# Imports
try:
    from cmfo.ontology.vocab import VocabularyManager
    from cmfo.storage.concept_store import ConceptStore
    from cmfo.cognition.axiom_mapper import AxiomMapper
    from cmfo.storage.bridge import KnowledgeBridge
except ImportError:
    pass

class ConceptDeriver:
    def __init__(self, domain: str):
        self.domain = domain
        self.vocab_mgr = VocabularyManager()
        self.vocab = self.vocab_mgr.get_context(domain)
        self.store = ConceptStore(domain)
        
        # Initialize Bridge
        try:
            bridge = KnowledgeBridge()
        except Exception:
            bridge = None
            print("Warning: KnowledgeBridge could not be loaded.")
            
        self.mapper = AxiomMapper(bridge=bridge)
        
        if not self.vocab:
            print(f"Warning: Domain {domain} not loaded properly.")

    def process_shard(self, shard_path: Path):
        """Streaming processing of a science shard"""
        print(f"Deriving {self.domain} concepts from {shard_path.name}...")
        
        count_candidates = 0
        count_stored = 0
        
        with open(shard_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    paper_id = record.get("id")
                    layers = record.get("layers", {})
                    definitions = layers.get("definitions", [])
                    theorems = layers.get("theorems", [])
                    proofs = layers.get("proofs", [])
                    
                    # 1. Process Definitions (Concepts)
                    for definition in definitions:
                        count_candidates += 1
                        term = self._extract_term_heuristic(definition)
                        if not term: continue
                        if not self.vocab.is_valid(term): continue
                        
                        raw_vec = self.mapper.formal_to_vector(definition)
                        if not raw_vec: continue
                        
                        final_vec = self.vocab.get_projection(raw_vec)
                        self.store.add_concept(term, definition, final_vec, paper_id, c_type="concept")
                        count_stored += 1

                    # 2. Process Theorems (Laws)
                    for thm in theorems:
                        # Extract "Theorem 1.2" or title
                        # Heuristic: Use first 50 chars as "Term" if explicit title absent
                        term = f"Theorem (Paper {paper_id})"
                        # Try to find ID
                        # "Theorem 3.1. Let..."
                        import re
                        m = re.match(r"(Theorem|Thm\.?)\s*(\d+(?:\.\d+)*)", thm)
                        if m:
                            term = f"{m.group(1)} {m.group(2)}"
                        
                        # Theorems still need vectors? Yes, derivation vectors.
                        raw_vec = self.mapper.formal_to_vector(thm)
                        if raw_vec:
                            final_vec = self.vocab.get_projection(raw_vec)
                            self.store.add_concept(term, thm, final_vec, paper_id, c_type="theorem")
                            count_stored += 1

                    # 3. Process Proofs
                    for prf in proofs:
                        target = "Proof"
                        # Ideally link to theorem, but for now just store content
                        self.store.add_concept(f"Proof ({paper_id})", prf, [], paper_id, c_type="proof")
                        count_stored += 1
                        
                except Exception as e:
                    continue
                    
        print(f"Derivation Complete.")
        print(f"Candidates: {count_candidates}")
        print(f"Stored:     {count_stored}")

    def _extract_term_heuristic(self, def_text: str) -> str:
        """
        Extracts 'X' from 'We define X as...'
        Very naive regex for v1.
        """
        import re
        # "We define the Turing Machine as..."
        match = re.search(r"define (?:the )?([a-zA-Z0-9\-\s]+?) as", def_text, re.IGNORECASE)
        if match:
             return match.group(1).strip()
             
        # "Let X be..."
        match = re.search(r"Let ([a-zA-Z0-9\-\s]+?) be", def_text, re.IGNORECASE)
        if match:
            t = match.group(1).strip()
            if len(t) < 50: return t
            
        return ""

if __name__ == "__main__":
    # Test Run
    pass
