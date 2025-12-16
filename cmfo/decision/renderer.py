"""
CMFO D18: The Pedagogical Renderer
==================================
The Voice of the System.
Translates algebraic structures and graph paths into human-readable explanations.
Strictly deterministic. No formatting hallucinations.
"""

from typing import List, Dict, Optional, Any
import random

# D16 imports
try:
    from cmfo.cognition.reasoner import ProofStep
    from cmfo.cognition.graph import ConceptNode
except ImportError:
    pass

class ExplanationRequest:
    def __init__(self, subject: str, audience: str = "student", depth: float = 0.5):
        self.subject = subject
        self.audience = audience
        self.depth = depth # 0.0 = Simple, 1.0 = Formal

class StructuralRenderer:
    def __init__(self, graph=None):
        self.graph = graph

    def explain_concept(self, term: str, depth: float = 0.5) -> str:
        """
        Explains a static concept or theorem.
        """
        if not self.graph:
            return f"Error: No knowledge graph attached. Cannot explain '{term}'."
            
        node = self.graph.get_node(term)
        if not node:
            return f"I do not have a formal definition for '{term}'."

        # 1. Identity (What is it?)
        definition = node.definition
        c_type = getattr(node, 'c_type', 'concept')
        
        # Clean up definition (remove artifacts if needed)
        clean_def = definition.strip()
        if len(clean_def) > 300 and depth < 0.5:
            clean_def = clean_def[:200] + "..."

        # 2. Context (What is it related to?)
        relations = []
        if depth > 0.3:
            # Fetch immediate neighbors
            for rel, target in node.edges_out[:3]: # Limit to 3
                relations.append(self._humanize_relation(term, rel, target))

        # 3. Assemble Output
        buffer = []
        
        # Header
        if c_type == 'theorem':
            buffer.append(f"🏛️ **Formal Law**: {term}")
        elif c_type == 'proof':
            buffer.append(f"📜 **Derivation**: {term}")
        else:
            buffer.append(f"💡 **Concept**: {term}")

        # Body
        buffer.append(f"\n**Definition**: {clean_def}")
        
        # Relations
        if relations and depth > 0.2:
            buffer.append("\n**Context**:")
            for r in relations:
                buffer.append(f"- {r}")
                
        # Vector Insight (Expert Only)
        if depth > 0.8 and node.vector:
            buffer.append(f"\n**Algebraic Signature**: {node.vector[:3]}... (7D projection)")

        return "\n".join(buffer)

    def explain_proof(self, trace_or_steps: Any, audit_result: Optional[Any] = None) -> str:
        """
        Explains a logical derivation path, with optional audit Context.
        Accepts ProofTrace object or list of steps.
        """
        # Unwrap trace if needed
        if hasattr(trace_or_steps, 'steps'):
            proof = trace_or_steps.steps
        else:
            proof = trace_or_steps
            
        if not proof:
            return "No derivation path found."

        buffer = []
        
        # 0. Metacognitive Header (D19 Integration)
        if audit_result:
            confidence = audit_result.confidence
            if confidence > 0.8:
                buffer.append("✅ **Scientifically Verified Derivation**")
                buffer.append(f"*(Confidence: {confidence:.2f} | Evidence: {audit_result.evidence_score:.2f})*")
            elif confidence > 0.5:
                buffer.append("⚠️ **Hypothetical Connection**")
                buffer.append(f"*(Confidence: {confidence:.2f} | Note: Plausible but lacks formal theorem support)*")
            else:
                buffer.append("❌ **Speculative Association**")
                buffer.append(f"*(Confidence: {confidence:.2f} | Warning: Logic gap detected)*")
            
            # Critical Flags
            if audit_result.flags.get("illegal_domain"):
                buffer.append("🚨 **Ontological Violation**: Terms cross sovereign domains without bridge.")
                
            buffer.append("") # Spacer

        buffer.append("**Trace of Reasoning:**")
        
        for i, step in enumerate(proof):
            # step has .description and .formal_rel
            
            # Simple Style
            buffer.append(f"{i+1}. {step.description}")
            
        return "\n".join(buffer)

    def _humanize_relation(self, src: str, rel: str, dst: str) -> str:
        """Maps formal edges to natural language."""
        if rel == "IS_A":
            return f"It is a type of **{dst}**."
        elif rel == "RELATES_TO":
            return f"It is conceptually linked to **{dst}**."
        elif rel == "IMPLIES":
            return f"It implies the existence of **{dst}**."
        return f"{rel} -> {dst}"
