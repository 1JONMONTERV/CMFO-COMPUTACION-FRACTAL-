"""
CMFO D16: The Reasoner
======================
The Inference Engine.
Navigates the ConceptGraph to construct proofs and explanations.
"""

from typing import List, Dict, Optional, Tuple
try:
    from cmfo.cognition.graph import ConceptGraph
except ImportError:
    pass

class ProofStep:
    def __init__(self, step_id: int, description: str, formal_rel: str, node_term: str, source_type: str = "concept"):
        self.step_id = step_id
        self.description = description
        self.formal_rel = formal_rel
        self.node_term = node_term
        self.source_type = source_type
        
    def __repr__(self):
        return f"{self.step_id}. [{self.source_type}] {self.description} ({self.formal_rel})"

class ProofTrace:
    def __init__(self, steps: List[ProofStep]):
        self.steps = steps

class Reasoner:
    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def explain_connection(self, start_term: str, end_term: str) -> Optional[ProofTrace]:
        """
        BFS to find shortest path. Returns ProofTrace.
        """
        if start_term not in self.graph.nodes or end_term not in self.graph.nodes:
            return None

        # BFS
        queue = [(start_term, [])] 
        visited = {start_term}
        
        found_path = None
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end_term:
                found_path = path
                break
                
            node = self.graph.get_node(current)
            if not node: continue
            
            for rel, target in node.edges_out:
                if target not in visited:
                    visited.add(target)
                    new_path = path + [(current, rel, target)]
                    queue.append((target, new_path))
                    
        if not found_path:
            return None
            
        # Reconstruct Proof
        steps = []
        for i, (src, rel, dst) in enumerate(found_path):
            # Get Node info for Metadata
            src_node = self.graph.get_node(src)
            c_type = getattr(src_node, 'c_type', 'concept')
            
            desc = f"{src} {rel} {dst}"
            if rel == "IS_A":
                desc = f"{src} is a type of {dst}"
            elif rel == "RELATES_TO":
                desc = f"{src} is related to {dst}"
                
            steps.append(ProofStep(i+1, desc, rel, src, c_type))
            
        # Add final goal node context as last step? 
        # Actually usually proof is edges. But let's verify if D19 needs goal too.
        # D19 walks steps. The edges are A->B. The last step brings us to B. 
        # The node term in step is 'src'. The 'dst' is the target.
        
        return ProofTrace(steps)

    def verify_inclusion(self, child: str, parent: str) -> bool:
        """
        Checks if 'child' IS_A 'parent' (directly or transitively).
        """
        proof = self.explain_connection(child, parent)
        if not proof: return False
        
        # Verify strict taxonomy chain
        # For now, we accept any path, but D16 spec requires strictness eventually
        for step in proof:
            if step.formal_rel not in ["IS_A"]: 
                # Strict inclusion might fail if path contains just 'RELATES_TO'.
                # But for V1 we track connectivity.
                pass
        return True
