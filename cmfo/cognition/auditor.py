"""
CMFO D19: The Metacognitive Auditor
===================================
Self-reflection layer that judges the quality of reasoning chains.
Calculates Solidity, Integrity, and Evidence scores to assign a Global Confidence.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class VerificationResult:
    confidence: float              # 0.0 - 1.0 (min(S,D,E))
    solidity_score: float          # S
    domain_integrity_score: float  # D
    evidence_score: float          # E
    issues: List[str]              # Warnings
    flags: Dict[str, bool]         # Critical flags

class MetacognitiveAuditor:
    def __init__(self, vocabulary_manager=None):
        self.vocab_mgr = vocabulary_manager

    def audit(self, proof_trace, target_domain: str = "computacion") -> VerificationResult:
        """
        Audits a ProofTrace object (list of steps + metadata).
        """
        issues = []
        
        # 1. Calculate Metrics
        S = self._calc_solidity(proof_trace, issues)
        D = self._calc_domain_integrity(proof_trace, target_domain, issues)
        E = self._calc_evidence_strength(proof_trace, issues)
        
        # 2. Global Confidence (Conservative)
        confidence = min(S, D, E)
        
        # 3. Raise Flags
        flags = {
            "speculative": confidence < 0.5,
            "illegal_domain": D < 0.7,
            "weak_evidence": E < 0.6,
            "formal_gap": S < 0.6,
            "circular": self._check_circularity(proof_trace)
        }
        
        if flags["circular"]:
            issues.append("Logical Loop Detected.")
            confidence = 0.0 # Circular logic is fatal

        return VerificationResult(
            confidence=confidence,
            solidity_score=S,
            domain_integrity_score=D,
            evidence_score=E,
            issues=issues,
            flags=flags
        )

    def _calc_solidity(self, trace, issues: List[str]) -> float:
        """
        M1 - Solidity (S): Logical connectivity.
        S = formal_steps / total_steps
        """
        steps = trace.steps
        if not steps: 
            return 0.0
            
        formal_count = 0
        for i, step in enumerate(steps):
            # Check edge type
            rel = step.formal_rel
            if rel in ["IS_A", "IMPLIES", "EQUIVALENT_TO"]:
                formal_count += 1
            elif rel == "RELATES_TO":
                # Weak association
                pass 
            # Treat Theorem usage as formal
            if "Theorem" in step.description or "Lemma" in step.description:
                formal_count += 1
                
        S = formal_count / len(steps)
        if S < 0.6:
            issues.append(f"Low Solidity ({S:.2f}): Heavy reliance on loose associations.")
        return S

    def _calc_domain_integrity(self, trace, domain: str, issues: List[str]) -> float:
        """
        M2 - Domain Integrity (D): Respect for ontology.
        D = 1.0 - (violations / steps)
        """
        steps = trace.steps
        if not steps: return 1.0
        
        violations = 0
        # If we have a vocab manager, use it. Otherwise assume naive check.
        if self.vocab_mgr:
            for step in steps:
                term = step.node_term
                if not self.vocab_mgr.is_valid_in_domain(term, domain):
                    violations += 1
                    issues.append(f"Domain Violation: '{term}' not valid in {domain}")
                    
        D = 1.0 - (violations / len(steps))
        return max(0.0, D)

    def _calc_evidence_strength(self, trace, issues: List[str]) -> float:
        """
        M3 - Evidence Strength (E): Weight of sources.
        """
        steps = trace.steps
        if not steps: return 0.0
        
        total_weight = 0.0
        
        weights = {
            "theorem": 1.0,
            "lemma": 1.0,
            "proof": 1.0,
            "definition": 0.8,
            "concept": 0.6, # Standard link
            "heuristic": 0.3
        }
        
        for step in steps:
            w = weights.get(step.source_type, 0.5) # Default 0.5
            total_weight += w
            
        E = total_weight / len(steps)
        if E < 0.6:
            issues.append(f"Weak Evidence ({E:.2f}): Few theorems/definitions in path.")
        return E
        
    def _check_circularity(self, trace) -> bool:
        visited = set()
        for step in trace.steps:
            if step.node_term in visited:
                return True
            visited.add(step.node_term)
        return False
