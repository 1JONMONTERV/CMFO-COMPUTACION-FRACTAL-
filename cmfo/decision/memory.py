"""
CMFO D5: Fractal Persistent Memory
===================================
Hierarchical memory system:
- Short-term: Recent conversation context (in-memory)
- Long-term: Persistent dreams (append-only file)

APPEND-ONLY, AUDITABLE, GEOMETRIC INDEXING.
"""

import json
import time
import hashlib
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MemoryEntry:
    """
    Single memory entry (dream).
    
    Append-only, never modified.
    """
    id: str  # M:0001342
    state_vector: List[float]  # 7D semantic state
    intent: str  # confirm, correct, question, reference, conflict
    proof_ref: str  # P:009812 (reference to proof object)
    context_hash: str  # Hash of context sources
    timestamp: int  # Unix timestamp
    confidence: float  # Margin stability (0-1)
    metadata: Dict = None  # Optional metadata
    
    def to_dict(self) -> Dict:
        """Export as dict"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        """Load from dict"""
        return cls(**data)


class DreamStore:
    """
    Persistent memory storage.
    
    Append-only JSON Lines format:
    - One entry per line
    - Never delete or modify
    - Only append new entries
    """
    
    def __init__(self, dream_file: str = "dreams.jsonl"):
        self.dream_file = Path(dream_file)
        self.entries: List[MemoryEntry] = []
        self.index = {}  # id -> entry
        
        # Create file if doesn't exist
        if not self.dream_file.exists():
            self.dream_file.touch()
        
        # Load existing dreams
        self._load_dreams()
    
    def _load_dreams(self):
        """Load all dreams from file"""
        if not self.dream_file.exists():
            return
        
        with open(self.dream_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = MemoryEntry.from_dict(json.loads(line))
                    self.entries.append(entry)
                    self.index[entry.id] = entry
    
    def store(self, entry: MemoryEntry):
        """
        Store new memory entry.
        
        APPEND-ONLY: Never modifies existing entries.
        """
        # Add to memory
        self.entries.append(entry)
        self.index[entry.id] = entry
        
        # Append to file
        with open(self.dream_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')
    
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get entry by ID"""
        return self.index.get(memory_id)
    
    def recent(self, n: int = 10) -> List[MemoryEntry]:
        """Get n most recent entries"""
        return self.entries[-n:]
    
    def count(self) -> int:
        """Total number of entries"""
        return len(self.entries)
    
    def __len__(self):
        return len(self.entries)


class FractalMemory:
    """
    Hierarchical fractal memory system.
    
    Short-term: Recent context (last N entries, in-memory)
    Long-term: All dreams (persistent, geometric index)
    """
    
    def __init__(self, 
                 dream_file: str = "dreams.jsonl",
                 short_term_size: int = 10):
        self.dream_store = DreamStore(dream_file)
        self.short_term_size = short_term_size
        self.next_id = len(self.dream_store) + 1
        self.next_proof_id = 1
    
    def store(self,
              state_vector: List[float],
              intent: str,
              proof_object: Dict,
              context_sources: List[str],
              confidence: float) -> str:
        """
        Store new memory entry.
        
        Args:
            state_vector: 7D semantic state
            intent: Decision intent
            proof_object: Complete proof object (dict)
            context_sources: List of context sources
            confidence: Margin stability (0-1)
            
        Returns:
            memory_id: ID of stored entry
        """
        # Generate IDs
        memory_id = f"M:{self.next_id:07d}"
        proof_id = f"P:{self.next_proof_id:06d}"
        
        # Hash context
        context_str = "|".join(sorted(context_sources))
        context_hash = hashlib.md5(context_str.encode()).hexdigest()[:12]
        
        # Create entry
        entry = MemoryEntry(
            id=memory_id,
            state_vector=state_vector,
            intent=intent,
            proof_ref=proof_id,
            context_hash=context_hash,
            timestamp=int(time.time()),
            confidence=confidence,
            metadata={
                "proof": proof_object,
                "context_sources": context_sources
            }
        )
        
        # Store
        self.dream_store.store(entry)
        
        # Increment counters
        self.next_id += 1
        self.next_proof_id += 1
        
        return memory_id
    
    def recall(self,
               query_state: List[float],
               k: int = 5,
               intent_filter: Optional[str] = None) -> List[Tuple[MemoryEntry, float]]:
        """
        Recall k most similar memories.
        
        Args:
            query_state: Query vector
            k: Number of results
            intent_filter: Optional intent filter
            
        Returns:
            List of (entry, distance) tuples, sorted by distance
        """
        # Import d_phi
        try:
            from .enhanced_engine import d_phi
        except ImportError:
            # Standalone execution
            import math
            PHI = 1.6180339887
            def d_phi(x: List[float], y: List[float]) -> float:
                dist_sq = 0.0
                for i in range(7):
                    weight = PHI ** i
                    diff = x[i] - y[i]
                    dist_sq += weight * diff * diff
                return math.sqrt(dist_sq)
        
        # Calculate distances
        candidates = []
        for entry in self.dream_store.entries:
            # Filter by intent if specified
            if intent_filter and entry.intent != intent_filter:
                continue
            
            dist = d_phi(query_state, entry.state_vector)
            candidates.append((entry, dist))
        
        # Sort by distance
        candidates.sort(key=lambda x: x[1])
        
        # Return top k
        return candidates[:k]
    
    def short_term(self) -> List[MemoryEntry]:
        """Get short-term memory (recent context)"""
        return self.dream_store.recent(self.short_term_size)
    
    def long_term(self) -> List[MemoryEntry]:
        """Get all long-term memory"""
        return self.dream_store.entries
    
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get specific memory by ID"""
        return self.dream_store.get(memory_id)
    
    def forget(self, policy: str = "none"):
        """
        Forget policy (non-destructive).
        
        Currently just a placeholder - could implement:
        - Deprioritization by age
        - Confidence-based filtering
        - Semantic clustering
        
        NEVER destructive - just changes recall weights.
        """
        # TODO: Implement forgetting policies
        pass
    
    def state_description(self) -> str:
        """Describe memory state"""
        total = len(self.dream_store)
        if total == 0:
            return "empty"
        elif total < 10:
            return "sparse"
        elif total < 100:
            return "moderate"
        else:
            return "rich"
    
    def stats(self) -> Dict:
        """Memory statistics"""
        entries = self.dream_store.entries
        
        if not entries:
            return {
                "total": 0,
                "by_intent": {},
                "avg_confidence": 0.0
            }
        
        # Count by intent
        by_intent = {}
        for entry in entries:
            by_intent[entry.intent] = by_intent.get(entry.intent, 0) + 1
        
        # Average confidence
        avg_conf = sum(e.confidence for e in entries) / len(entries)
        
        return {
            "total": len(entries),
            "by_intent": by_intent,
            "avg_confidence": round(avg_conf, 3),
            "oldest": entries[0].timestamp if entries else None,
            "newest": entries[-1].timestamp if entries else None
        }
    
    def __len__(self):
        """Return total number of entries"""
        return len(self.dream_store)


if __name__ == "__main__":
    print("CMFO D5: Fractal Persistent Memory")
    print("=" * 60)
    
    # Test
    memory = FractalMemory(dream_file="test_dreams.jsonl")
    
    # Store some entries
    print("\nStoring memories...")
    
    for i in range(5):
        memory_id = memory.store(
            state_vector=[0.8 - i*0.1, 0.2 + i*0.05, 0.1, 0.0, 0.0, 0.0, 0.0],
            intent="confirm" if i % 2 == 0 else "correct",
            proof_object={"test": f"proof_{i}"},
            context_sources=[f"doc_{i}.txt"],
            confidence=0.8 + i*0.02
        )
        print(f"  Stored: {memory_id}")
    
    # Recall
    print("\nRecalling similar memories...")
    query = [0.75, 0.25, 0.1, 0.0, 0.0, 0.0, 0.0]
    results = memory.recall(query, k=3)
    
    for entry, dist in results:
        print(f"  {entry.id}: d_phi={dist:.4f}, intent={entry.intent}, conf={entry.confidence}")
    
    # Stats
    print("\nMemory Statistics:")
    stats = memory.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Short-term
    print("\nShort-term memory:")
    for entry in memory.short_term():
        print(f"  {entry.id}: {entry.intent}")
    
    print("\nMemory system loaded successfully.")
    
    # Cleanup test file
    import os
    if os.path.exists("test_dreams.jsonl"):
        os.remove("test_dreams.jsonl")
