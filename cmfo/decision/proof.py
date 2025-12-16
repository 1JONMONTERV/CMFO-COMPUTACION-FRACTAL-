"""
CMFO Proof Object System (D2)
==============================
Every decision comes with complete audit trail.

No black box - full geometric evidence.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class EvidenceType(Enum):
    """Types of evidence supporting a decision"""
    MEMORY_HIT = "memory_hit"
    RULE_CHECK = "rule"
    CONTEXT_MATCH = "context"
    GEOMETRIC_DISTANCE = "distance"
    THRESHOLD_TEST = "threshold"


@dataclass
class Evidence:
    """Single piece of evidence"""
    type: EvidenceType
    data: Dict[str, Any]
    weight: float = 1.0
    
    def __repr__(self):
        return f"Evidence({self.type.value}: {self.data})"


@dataclass
class Candidate:
    """Decision candidate with score breakdown"""
    label: str
    type: str
    score: float
    d_input: float
    d_memory: float = 0.0
    d_context: float = 0.0
    cost: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "type": self.type,
            "score": round(self.score, 4),
            "d_input": round(self.d_input, 4),
            "d_memory": round(self.d_memory, 4),
            "d_context": round(self.d_context, 4),
            "cost": round(self.cost, 2)
        }


@dataclass
class ProofObject:
    """
    Complete proof of decision.
    
    This is the CORE of auditable AI:
    - What was decided
    - Why it was decided
    - What evidence supports it
    - What alternatives were considered
    - What thresholds were used
    """
    # Decision
    intent: str  # 'confirm', 'correct', 'question', 'reference', 'conflict'
    winner: Candidate
    runner_up: Optional[Candidate] = None
    
    # Margins (stability)
    delta: float = 0.0  # Score difference (winner - runner_up)
    margin_stable: bool = True  # delta > tau_uncertain
    
    # Evidence trail
    evidence: List[Evidence] = field(default_factory=list)
    
    # Thresholds used
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Context
    input_classification: str = ""  # 'correct', 'false', 'ambiguous', etc.
    memory_state: str = ""  # 'empty', 'sparse', 'rich'
    
    def to_dict(self) -> Dict:
        """Export as JSON-serializable dict"""
        return {
            "intent": self.intent,
            "winner": self.winner.to_dict(),
            "runner_up": self.runner_up.to_dict() if self.runner_up else None,
            "margins": {
                "delta": round(self.delta, 4),
                "stable": self.margin_stable
            },
            "evidence": [
                {
                    "type": e.type.value,
                    "data": e.data,
                    "weight": e.weight
                }
                for e in self.evidence
            ],
            "thresholds": self.thresholds,
            "context": {
                "input_classification": self.input_classification,
                "memory_state": self.memory_state
            }
        }
    
    def explain(self) -> str:
        """
        Human-readable explanation.
        
        This is NOT the final response - it's the technical explanation
        of WHY the decision was made.
        """
        lines = []
        lines.append(f"Decision: {self.intent} ({self.winner.label})")
        lines.append(f"Confidence: {'HIGH' if self.margin_stable else 'LOW'} (delta={self.delta:.3f})")
        lines.append(f"")
        lines.append(f"Scores:")
        lines.append(f"  Winner: {self.winner.label} = {self.winner.score:.3f}")
        if self.runner_up:
            lines.append(f"  Runner-up: {self.runner_up.label} = {self.runner_up.score:.3f}")
        lines.append(f"")
        lines.append(f"Evidence ({len(self.evidence)} items):")
        for i, e in enumerate(self.evidence[:5], 1):  # Show top 5
            lines.append(f"  {i}. {e.type.value}: {e.data}")
        
        return "\n".join(lines)


class ProofBuilder:
    """Builds proof objects from decision process"""
    
    def __init__(self, tau_uncertain: float = 0.05, tau_certain: float = 0.15):
        self.tau_uncertain = tau_uncertain
        self.tau_certain = tau_certain
    
    def build(self,
              winner: Candidate,
              runner_up: Optional[Candidate],
              evidence: List[Evidence],
              input_classification: str,
              memory_state: str) -> ProofObject:
        """
        Build complete proof object.
        
        Args:
            winner: Winning candidate
            runner_up: Second-place candidate
            evidence: List of evidence items
            input_classification: Type of input
            memory_state: State of memory
            
        Returns:
            Complete ProofObject
        """
        # Calculate margin
        delta = 0.0
        if runner_up:
            delta = abs(runner_up.score - winner.score)
        
        # Stability check
        margin_stable = delta >= self.tau_uncertain
        
        # Build thresholds dict
        thresholds = {
            "tau_uncertain": self.tau_uncertain,
            "tau_certain": self.tau_certain,
            "tau_confirm": 0.18,  # Specific for confirm
            "tau_correction": 0.35,  # Specific for correction
            "tau_conflict": 0.95  # Specific for conflict
        }
        
        return ProofObject(
            intent=winner.type,
            winner=winner,
            runner_up=runner_up,
            delta=delta,
            margin_stable=margin_stable,
            evidence=evidence,
            thresholds=thresholds,
            input_classification=input_classification,
            memory_state=memory_state
        )


if __name__ == "__main__":
    # Test
    print("CMFO Proof Object System")
    print("=" * 60)
    
    # Example proof
    winner = Candidate(
        label="confirmation",
        type="confirm",
        score=0.312,
        d_input=0.1272,
        d_memory=0.05,
        cost=0.1
    )
    
    runner_up = Candidate(
        label="question",
        type="question",
        score=0.330,
        d_input=0.25,
        cost=0.3
    )
    
    evidence = [
        Evidence(
            type=EvidenceType.MEMORY_HIT,
            data={"item_id": "M:1342", "d_phi": 0.09}
        ),
        Evidence(
            type=EvidenceType.RULE_CHECK,
            data={"name": "L3_non_commutativity", "passed": True}
        )
    ]
    
    builder = ProofBuilder()
    proof = builder.build(
        winner=winner,
        runner_up=runner_up,
        evidence=evidence,
        input_classification="correct",
        memory_state="sparse"
    )
    
    print("\nProof Object:")
    print(proof.explain())
    print("\nJSON Export:")
    import json
    print(json.dumps(proof.to_dict(), indent=2))
    
    print("\nProof system loaded successfully.")
