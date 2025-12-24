"""
CMFO D16: Concept Graph
=======================
The In-Memory Knowledge Representation.
Loads concepts from storage and builds a queryable graph structure.
"""

import sqlite3
import json
from typing import Dict, List, Set, Optional, Tuple

class ConceptNode:
    def __init__(self, term: str, vector: List[float], definition: str, source: str, c_type: str = "concept"):
        self.term = term
        self.vector = vector
        self.definition = definition
        self.source = source
        self.c_type = c_type
        self.edges_out: List[Tuple[str, str]] = [] # (relation_type, target_term)
        self.edges_in: List[Tuple[str, str]] = []

class ConceptGraph:
    def __init__(self, domain: str, db_path: str):
        self.domain = domain
        self.db_path = db_path
        self.nodes: Dict[str, ConceptNode] = {}
        self.loaded = False

    def load(self):
        """Hydrates graph from SQLite"""
        print(f"Hydrating Graph for {self.domain}...")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        try:
            c.execute("SELECT * FROM concepts")
            rows = c.fetchall()
            
            for r in rows:
                term = r["term"]
                try:
                    vec = json.loads(r["vector_json"])
                except:
                    vec = []
                
                # Handle missing concept_type for backward compatibility if migration failed
                c_type = r["concept_type"] if "concept_type" in r.keys() else "concept"
                
                node = ConceptNode(term, vec, r["definition_formal"], r["provenance"], c_type)
                self.nodes[term] = node
                
            self.loaded = True
            print(f"Graph Hydrated: {len(self.nodes)} nodes.")
            
            # Phase 2: Infer Edges (Naive Linkage for D16 Start)
            self._infer_edges()
            
        except Exception as e:
            print(f"Graph Load Error: {e}")
        finally:
            conn.close()

    def _infer_edges(self):
        """
        Builds graph edges based on internal references.
        If definition of A contains term B, add A -> B edge.
        """
        count_edges = 0
        for term, node in self.nodes.items():
            def_lower = node.definition.lower()
            
            # Complexity: O(N^2) naive, optimization needed for production
            # For 1k concepts it's fine.
            for potential_target in self.nodes:
                if term == potential_target: continue
                
                # If definition mentions another concept
                if f" {potential_target.lower()} " in f" {def_lower} ":
                    # Determine relation type (heuristic)
                    rel = "RELATES_TO"
                    if "is a" in def_lower or "defined as a" in def_lower:
                        rel = "IS_A"
                    
                    self.add_edge(term, potential_target, rel)
                    count_edges += 1
                    
        print(f"Inferred Edges: {count_edges}")

    def add_edge(self, source: str, target: str, rel_type: str):
        if source in self.nodes and target in self.nodes:
            self.nodes[source].edges_out.append((rel_type, target))
            self.nodes[target].edges_in.append((rel_type, source))

    def get_node(self, term: str) -> Optional[ConceptNode]:
        return self.nodes.get(term)
