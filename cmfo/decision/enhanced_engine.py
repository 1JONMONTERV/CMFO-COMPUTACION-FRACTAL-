"""
CMFO Enhanced Decision Engine (D2+D3+D4+D5+D6)
===============================================
Geometric decision with:
- D2: Proof objects (audit trail)
- D3: Deterministic rendering
- D4: Multi-source context scoring
- D5: Fractal persistent memory
- D6: Calibrated attractors

Score(a) = α·d_φ(S_in, A_a) + β·d_φ(M, A_a) + γ·d_φ(C, A_a) + η·Penalty(a)
"""

import math
import json
from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from .proof import ProofObject, Candidate, Evidence, EvidenceType, ProofBuilder
    from .renderer import DeterministicRenderer
    from .memory import FractalMemory, MemoryEntry
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from proof import ProofObject, Candidate, Evidence, EvidenceType, ProofBuilder
    from renderer import DeterministicRenderer
    from memory import FractalMemory, MemoryEntry


PHI = 1.6180339887


@dataclass
class SemanticState:
    """Semantic state in T^7_φ (for backward compatibility)"""
    vector: List[float]
    label: str
    type: str


@dataclass
class Context:
    """Active context (document, web, selection, etc.)"""
    vectors: List[List[float]]  # Context semantic states
    sources: List[str]  # Source labels
    weights: List[float]  # Importance weights
    
    def average_vector(self) -> Optional[List[float]]:
        """Get weighted average context vector"""
        if not self.vectors:
            return None
        
        avg = [0.0] * 7
        total_weight = sum(self.weights)
        
        for vec, weight in zip(self.vectors, self.weights):
            for i in range(7):
                avg[i] += vec[i] * weight
        
        if total_weight > 0:
            for i in range(7):
                avg[i] /= total_weight
        
        return avg


def d_phi(x: List[float], y: List[float]) -> float:
    """Phi-weighted distance"""
    dist_sq = 0.0
    for i in range(7):
        weight = PHI ** i
        diff = x[i] - y[i]
        dist_sq += weight * diff * diff
    return math.sqrt(dist_sq)


def phi_norm(x: List[float]) -> float:
    """Phi-weighted norm"""
    return math.sqrt(sum(PHI ** i * x[i] ** 2 for i in range(7)))


class EnhancedDecisionEngine:
    """
    Enhanced decision engine with proof objects and multi-source scoring.
    
    D2: Every decision produces a ProofObject
    D3: ProofObject can be rendered to text
    D4: Scoring uses input + memory + context
    D5: Persistent memory with automatic storage and recall
    """
    
    def __init__(self,
                 memory: Optional[FractalMemory] = None,
                 alpha: float = 1.0,   # Weight for input similarity
                 beta: float = 0.5,    # Weight for memory coherence
                 gamma: float = 0.3,   # Weight for context match
                 eta: float = 0.2,     # Weight for cost/penalty
                 citation_threshold: float = 0.15):  # When to cite precedents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.citation_threshold = citation_threshold
        
        # D5: Fractal memory
        self.memory = memory if memory else FractalMemory()
        
        # D6: Fractal Encoder
        try:
            from .encoder import FractalEncoder
            self.encoder = FractalEncoder()
        except ImportError:
            # Fallback for local run
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from encoder import FractalEncoder
            self.encoder = FractalEncoder()

        self.response_templates = self._build_templates()
        self.proof_builder = ProofBuilder()
        self.renderer = DeterministicRenderer(language="es")
    
    def _build_templates(self) -> Dict[str, SemanticState]:
        """
        Build response templates (semantic attractors).
        
        Priority:
        1. Human-calibrated (attractors_human_v1.json)
        2. Synthetic-calibrated (attractors_v1.json)
        3. Defaults
        """
        # Try human-calibrated first
        human_file = Path("attractors_human_v1.json")
        if human_file.exists():
            try:
                with open(human_file, 'r', encoding='utf-8') as f:
                    calibration = json.load(f)
                
                templates = {}
                for intent, spec in calibration['attractors'].items():
                    templates[intent] = SemanticState(
                        vector=spec['centroid'],
                        label=spec['posture'],  # Human posture label
                        type=intent
                    )
                
                print(f"[D6] Loaded HUMAN-calibrated attractors v{calibration['version']}")
                print(f"[D6] Method: {calibration['calibration_method']}")
                return templates
            except Exception as e:
                print(f"[D6] Warning: Failed to load human attractors ({e})")
        
        # Try synthetic-calibrated
        synthetic_file = Path("attractors_v1.json")
        if synthetic_file.exists():
            try:
                with open(synthetic_file, 'r', encoding='utf-8') as f:
                    calibration = json.load(f)
                
                templates = {}
                for intent, spec in calibration['attractors'].items():
                    templates[intent] = SemanticState(
                        vector=spec['centroid'],
                        label=spec['intent'],
                        type=intent
                    )
                
                print(f"[D6] Loaded synthetic-calibrated attractors v{calibration['version']}")
                return templates
            except Exception as e:
                print(f"[D6] Warning: Failed to load synthetic attractors ({e})")
        
        # Fallback to defaults
        print("[D6] Using default attractors (no calibration file)")
        return {
            'confirm': SemanticState(
                vector=[0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
                label='confirmation',
                type='confirm'
            ),
            'correct': SemanticState(
                vector=[-0.8, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0],
                label='correction',
                type='correct'
            ),
            'question': SemanticState(
                vector=[0.0, 0.0, 0.8, 0.5, 0.2, 0.0, 0.0],
                label='question',
                type='question'
            ),
            'reference': SemanticState(
                vector=[0.5, 0.5, 0.0, 0.0, 0.5, 0.3, 0.0],
                label='memory_reference',
                type='reference'
            ),
            'conflict': SemanticState(
                vector=[-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                label='contradiction_signal',
                type='conflict'
            )
        }
    
    def cost(self, state: SemanticState) -> float:
        """Cost/penalty for producing this response"""
        costs = {
            'confirm': 0.1,
            'correct': 0.5,
            'question': 0.3,
            'reference': 0.4,
            'conflict': 0.8
        }
        return costs.get(state.type, 1.0)
    
    def decide(self, 
               input_data: Union[List[float], str], 
               context: Optional[Context] = None, 
               slots: Optional[Dict] = None) -> Tuple[str, ProofObject]:
        """
        Make a geometric decision based on input, memory, and context.
        
        Args:
            input_data: 7D vector OR text string (auto-encoded)
            context: Optional context from other sources
            slots: Optional values for template filling
            
        Returns:
            (natural_language_response, proof_object)
        """
        # 1. Encode Input (D6 Industrial Standard)
        S_in = self._encode_input(input_data)
        
        # 2. Score Candidates against Input (D1)
        scored_candidates, memory_recalls = self._score_candidates(S_in, context)
        
        # 3. Collect Evidence (D4)
        winner = scored_candidates[0]
        runner_up = scored_candidates[1] if len(scored_candidates) > 1 else None
        
        # Calculate memory & context evidence
        evidence = self._collect_evidence(S_in, context, winner, memory_recalls)
        
        # Classify input
        input_class = self._classify_input(S_in)

        # 4. Build Proof Object (D2)
        proof = self.proof_builder.build(
            winner=winner, 
            runner_up=runner_up, 
            evidence=evidence, 
            input_classification=input_class,
            memory_state=self.memory.state_description()
        )
        
        # 5. Store in Fractal Memory (D5)
        # Only store significant interactions (not silent confirms of confirms)
        context_sources = context.sources if context else []
        self.memory.store(
            state_vector=S_in,
            intent=proof.intent,
            proof_object=proof.to_dict(),
            context_sources=context_sources,
            confidence=1.0 if proof.margin_stable else 0.5
        )
        
        # 6. Render Response (D3)
        response = self._render_with_citation(proof, slots, memory_recalls)
            
        return response, proof

    def _encode_input(self, input_data: Union[List[float], str]) -> List[float]:
        """
        Encode input to 7D vector deterministically.
        Supports both direct vector (legacy/test) and text (production).
        """
        if isinstance(input_data, list):
            if len(input_data) != 7:
                # Pad or truncate
                return (input_data + [0.0]*7)[:7]
            return input_data
        
        if isinstance(input_data, str):
            # Use FractalEncoder
            try:
                return self.encoder.encode(input_data)
            except Exception as e:
                print(f"Warning: FractalEncoder failed ({e}), falling back to zero vector.")
                return [0.0] * 7
        
        return [0.0] * 7
    
    def _score_candidates(self, S_input: List[float], context: Optional[Context]) -> Tuple[List[Candidate], List[Tuple[MemoryEntry, float]]]:
        """
        Scores all response candidates based on input, memory, and context.
        Returns sorted candidates and memory recalls.
        """
        candidates = list(self.response_templates.values())
        
        # D5: Get memory average from recall (not just recent)
        memory_recalls = self.memory.recall(S_input, k=3)
        if memory_recalls:
            avg_memory = self._average_states([entry.state_vector for entry, _ in memory_recalls])
        else:
            avg_memory = None
        
        # Get context average
        avg_context = context.average_vector() if context else None
        
        # Score all candidates
        scored_candidates = []
        
        for candidate in candidates:
            # D4: Multi-source scoring
            d_input = d_phi(candidate.vector, S_input)
            d_memory = d_phi(candidate.vector, avg_memory) if avg_memory else 0.0
            d_context = d_phi(candidate.vector, avg_context) if avg_context else 0.0
            c = self.cost(candidate)
            
            # Total score (minimize)
            score = (self.alpha * d_input + 
                    self.beta * d_memory + 
                    self.gamma * d_context + 
                    self.eta * c)
            
            scored_candidates.append(Candidate(
                label=candidate.label,
                type=candidate.type,
                score=score,
                d_input=d_input,
                d_memory=d_memory,
                d_context=d_context,
                cost=c
            ))
        
        # Sort by score (lower is better)
        scored_candidates.sort(key=lambda x: x.score)
        
        winner = scored_candidates[0]
        runner_up = scored_candidates[1] if len(scored_candidates) > 1 else None
        
        # Build evidence (D5: includes memory recalls)
        evidence_list = self._collect_evidence(S_input, context, winner, memory_recalls)
        
        # Classify input
        input_class = self._classify_input(S_input)
        
        # D2: Build proof object (NOT FULL BUILD HERE, just partial for validation if needed, 
        # but actually score_candidates shouldn't build the proof, decide() does)
        # We just return the sorted candidates and memory recalls
        
        return scored_candidates, memory_recalls
    
    def _collect_evidence(self,
                         S_input: List[float],
                         context: Optional[Context],
                         winner: Candidate,
                         memory_recalls: List[Tuple]) -> List[Evidence]:
        """
        Collect evidence supporting the decision.
        
        Memory evidence only collected when it will be cited (CORRECT/CONFLICT).
        Dual thresholds: < 0.12 (same), 0.12-0.25 (related), > 0.25 (irrelevant)
        """
        evidence = []
        
        # Memory evidence (only if will be cited)
        if winner.type in ['correct', 'conflict']:
            for entry, dist in memory_recalls:
                # Dual threshold: only same or related experiences
                if dist < 0.25:
                    evidence.append(Evidence(
                        type=EvidenceType.MEMORY_HIT,
                        data={
                            "memory_id": entry.id,
                            "d_phi": round(dist, 4),
                            "intent": entry.intent,
                            "timestamp": entry.timestamp,
                            "relation": "same" if dist < 0.12 else "related"
                        },
                        weight=1.0 / (dist + 0.01)
                    ))
        
        # Context matches (always relevant)
        if context and context.vectors:
            for i, (vec, source) in enumerate(zip(context.vectors, context.sources)):
                dist = d_phi(S_input, vec)
                if dist < 0.2:
                    evidence.append(Evidence(
                        type=EvidenceType.CONTEXT_MATCH,
                        data={"source": source, "d_phi": round(dist, 4)},
                        weight=context.weights[i]
                    ))
        
        # Geometric distance (always included for audit)
        evidence.append(Evidence(
            type=EvidenceType.GEOMETRIC_DISTANCE,
            data={"d_input": round(winner.d_input, 4), "attractor": winner.label},
            weight=1.0
        ))
        
        return evidence
    
    def _render_with_citation(self,
                             proof: ProofObject,
                             slots: Optional[Dict],
                             memory_recalls: List[Tuple]) -> str:
        """
        Render with human-like memory integration.
        
        PRINCIPLE: Memory always influences, rarely mentioned.
        
        Citation policy (human-like):
        - CONFIRM: Never cite (doesn't add value)
        - REFERENCE: Optional, only if clarifies
        - QUESTION: Implicit (guides without mentioning)
        - CORRECT: Always cite (justification needed)
        - CONFLICT: Always cite (evidence required)
        
        Dual thresholds:
        - d_phi < 0.12: Same precedent (same experience)
        - 0.12-0.25: Related experience
        - > 0.25: Irrelevant (don't cite)
        """
        if slots is None:
            slots = {}
        
        # Check closest memory
        should_cite = False
        cite_mode = None  # 'same', 'related', or None
        
        if memory_recalls:
            closest_entry, closest_dist = memory_recalls[0]
            
            # Dual threshold classification
            if closest_dist < 0.12:
                cite_mode = 'same'
            elif closest_dist < 0.25:
                cite_mode = 'related'
            else:
                cite_mode = None  # Too far, irrelevant
            
            # Intent-based citation decision
            if proof.intent == "correct":
                # Corrections need justification
                should_cite = cite_mode in ['same', 'related']
            elif proof.intent == "conflict":
                # Conflicts need evidence
                should_cite = cite_mode in ['same', 'related']
            elif proof.intent == "reference":
                # Only cite if same precedent (clarifies)
                should_cite = cite_mode == 'same'
            # confirm and question: never cite explicitly
        
        # Frame citation as implicit experience (human-like)
        if should_cite and memory_recalls:
            closest_entry, closest_dist = memory_recalls[0]
            
            # NEVER say "según mi memoria" or "tengo X entradas"
            # Frame as implicit experience
            
            if cite_mode == 'same':
                # Same precedent - frame as continuity
                if proof.intent == "correct":
                    slots["evidence"] = "esto contradice lo anterior"
                elif proof.intent == "conflict":
                    slots["previous"] = "lo que vimos antes"
                    slots["explanation"] = "las conclusiones son opuestas"
            
            elif cite_mode == 'related':
                # Related experience - frame as pattern
                if proof.intent == "correct":
                    slots["evidence"] = "en casos similares, la conclusión fue distinta"
                elif proof.intent == "conflict":
                    slots["previous"] = "situaciones parecidas"
                    slots["explanation"] = "el patrón es inconsistente"
        
        # Render (memory influence is implicit in scoring, not in text)
        return self.renderer.render(proof, slots)
    
    def _classify_input(self, S_input: List[float]) -> str:
        """Classify input type (D5: uses memory attractors from FractalMemory)"""
        # Get all memory vectors as attractors
        all_entries = self.memory.long_term()
        if all_entries:
            attractors = [entry.state_vector for entry in all_entries]
            min_dist = min(d_phi(S_input, attr) for attr in attractors)
            if min_dist < 0.1:
                return 'repetition'
            elif min_dist > 1.0:
                return 'contradiction'
        
        norm = phi_norm(S_input)
        if norm < 0.5:
            return 'ambiguous'
        
        if S_input[0] > 0:
            return 'correct'
        else:
            return 'false'
    
    def _average_states(self, vectors: List[List[float]]) -> List[float]:
        """Average multiple vectors"""
        if not vectors:
            return [0.0] * 7
        
        avg = [0.0] * 7
        for vec in vectors:
            for i in range(7):
                avg[i] += vec[i]
        
        for i in range(7):
            avg[i] /= len(vectors)
        
        return avg


if __name__ == "__main__":
    print("CMFO Enhanced Decision Engine (D2+D3+D4+D5)")
    print("=" * 60)
    
    # Test with D5 memory
    memory = FractalMemory(dream_file="test_engine_dreams.jsonl")
    engine = EnhancedDecisionEngine(memory=memory)
    
    # Add context (e.g., from document)
    context = Context(
        vectors=[[0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]],
        sources=["documento_actual.txt"],
        weights=[1.0]
    )
    
    # Input
    S_input = [0.75, 0.25, 0.1, 0.0, 0.0, 0.0, 0.0]
    
    # Decide (will auto-store in memory)
    response, proof = engine.decide(S_input, context)
    
    print("\nInput:", S_input[:3], "...")
    print("\nResponse:", response)
    print("\nProof:")
    print(proof.explain())
    print("\nMemory stats:", memory.stats())
    
    # Make similar decision to test citation
    S_input_2 = [0.76, 0.24, 0.11, 0.0, 0.0, 0.0, 0.0]
    response_2, proof_2 = engine.decide(S_input_2, context)
    
    print("\n\n[Second Decision - Should Cite Precedent]")
    print("Input:", S_input_2[:3], "...")
    print("Response:", response_2)
    print("\nMemory stats:", memory.stats())
    
    print("\nEngine loaded successfully.")
    
    # Cleanup
    import os
    if os.path.exists("test_engine_dreams.jsonl"):
        os.remove("test_engine_dreams.jsonl")
