from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import datetime
import hashlib
import json

@dataclass(frozen=True)
class ActionReceipt:
    """
    Inmutable receipt of an executed action.
    Traceability: Who authorized it, why, and what was the result.
    """
    action_id: str
    timestamp: str
    requested_by: Dict[str, str]  # e.g. {"type": "human", "id": "user_123"}
    authorized_by: List[str]      # e.g. ["USER", "DOMAIN:MATH-10", "PROOF:p1"]
    blocked_by: List[str]         # If failed, who blocked it (e.g. ["LEGAL:medical"])
    domain: str
    action: str
    input_data: str
    proof_ref: Optional[str]      # Reference to the ProofObject
    result: str
    confidence: float
    receipt_hash: str = field(init=False)

    def __post_init__(self):
        # Calculate hash of the content to ensure immutability claim
        content = f"{self.action_id}{self.timestamp}{self.requested_by}{self.authorized_by}{self.domain}{self.action}{self.input_data}{self.result}"
        # Bypass frozen attribute to set hash
        object.__setattr__(self, 'receipt_hash', hashlib.sha256(content.encode('utf-8')).hexdigest())

    def to_json(self) -> str:
        data = {
            "action_id": self.action_id,
            "timestamp": self.timestamp,
            "requested_by": self.requested_by,
            "authorized_by": self.authorized_by,
            "blocked_by": self.blocked_by,
            "domain": self.domain,
            "action": self.action,
            "input": self.input_data,
            "proof_ref": self.proof_ref,
            "result": self.result,
            "confidence": self.confidence,
            "hash": self.receipt_hash
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
