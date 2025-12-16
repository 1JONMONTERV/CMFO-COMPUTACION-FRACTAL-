"""
CMFO D14: Vocabulary Manager (Ontological Sovereignity)
=======================================================
Manages domain-specific vocabularies, ensuring terms are evaluated
within their proper context (e.g. "Field" in Physics vs CS).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

VOCAB_ROOT = Path("D:/CMFO_DATA/vocabularies")

class Vocabulary:
    def __init__(self, domain: str, data: Dict[str, Any]):
        self.domain = domain
        self.terms = data.get("terms", {}) # map term -> properties
        self.axioms = data.get("axioms", [])
        self.allowed_ops = set(data.get("allowed_ops", []))
        self.forbidden = set(data.get("forbidden", []))
        self.modifiers = data.get("vector_modifiers", {}) # axis_index -> multiplier

    def is_valid(self, term: str) -> bool:
        if term in self.forbidden:
            return False
        # If strict mode, we might require term to be in self.terms
        # For now, we only check explicit prohibition
        return True 

    def get_projection(self, base_vector: List[float]) -> List[float]:
        """
        Project a generic 7D vector into the domain's sub-manifold.
        e.g. In Computation, "Mente" (Axis 5) might be zeroed.
        """
        if not base_vector or len(base_vector) != 7:
            return base_vector
            
        new_vec = list(base_vector)
        for axis_idx, mod in self.modifiers.items():
            try:
                idx = int(axis_idx)
                if 0 <= idx < 7:
                     new_vec[idx] *= mod
            except ValueError:
                pass
        return new_vec

class VocabularyManager:
    def __init__(self):
        self.loaded_domains: Dict[str, Vocabulary] = {}
        VOCAB_ROOT.mkdir(parents=True, exist_ok=True)

    def define_domain(self, domain: str, config: Dict):
        """Creates/Overwrites a domain definition on disk (JSONL)"""
        path = VOCAB_ROOT / f"{domain}.jsonl"
        
        with open(path, 'w', encoding='utf-8') as f:
            # Line 1: Metadata
            meta = {
                "type": "domain_meta",
                "domain": domain,
                "axioms": config.get("axioms", []),
                "allowed_ops": config.get("allowed_ops", []),
                "forbidden": config.get("forbidden", []),
                "vector_modifiers": config.get("vector_modifiers", {})
            }
            f.write(json.dumps(meta) + "\n")
            
            # Subsequent lines: Seed terms
            if "terms" in config and isinstance(config["terms"], list):
                for t in config["terms"]:
                    f.write(json.dumps(t) + "\n")
                    
        print(f"Defined Domain: {domain} at {path}")
        # Identify in loaded cache
        self.load_domain(domain)

    def load_domain(self, domain: str) -> Optional[Vocabulary]:
        path = VOCAB_ROOT / f"{domain}.jsonl"
        if not path.exists():
            return None
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                # Read meta
                line1 = f.readline()
                if not line1: return None
                
                meta = json.loads(line1)
                if meta.get("type") != "domain_meta":
                    return None 
                
                # Read terms
                terms = {}
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "term" in obj:
                            terms[obj["term"]] = obj
                    except json.JSONDecodeError:
                        continue
                
                # Construct
                data = meta
                data["terms"] = terms
                
                vocab = Vocabulary(domain, data)
                self.loaded_domains[domain] = vocab
                return vocab
                
        except Exception as e:
            print(f"Error loading {domain}: {e}")
            return None

    def get_context(self, domain: str) -> Optional[Vocabulary]:
        if domain in self.loaded_domains:
            return self.loaded_domains[domain]
        return self.load_domain(domain)
