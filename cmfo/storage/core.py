"""
CMFO D9: Storage Infrastructure
===============================
Industrial Storage Module for Massive Semantic Data (2TB+).

Implements the Split Schema:
1. LexicalStore (Source Definitions) -> Immutable Shards
2. VectorStore (Calculated Values) -> Versioned Shards
3. Index (O(1) Lookup) -> SQLite

Path layout: D:/CMFO_DATA/
"""

import os
import json
import sqlite3
import hashlib
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict

# Configuration
STORAGE_ROOT = Path("D:/CMFO_DATA")
SHARDS_DIR = STORAGE_ROOT / "shards" / "es"
VECTORS_DIR = STORAGE_ROOT / "vectors" / "v1"
INDEX_DIR = STORAGE_ROOT / "index"

# Ensure directories exist (redundant safety)
for d in [SHARDS_DIR, VECTORS_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class AlgebraicDef:
    term: str
    algebraic_def: Dict[str, List[str]]
    source: str = "unknown"
    lang: str = "es"

@dataclass
class VectorEntry:
    vector_id: str
    term: str
    vector: List[float]
    axiom_version: str
    phi_norm: float


class Index:
    """SQLite-based Index for O(1) Lookups"""
    
    def __init__(self):
        self.lex_db_path = INDEX_DIR / "lexeme.db"
        self.vec_db_path = INDEX_DIR / "vector.db"
        self._init_dbs()
        
    def _init_dbs(self):
        # Lexeme Index
        with sqlite3.connect(self.lex_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lexemes (
                    term TEXT PRIMARY KEY,
                    shard_id INTEGER,
                    offset INTEGER,
                    vector_id TEXT
                )
            """)
            
        # Vector Index
        with sqlite3.connect(self.vec_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id TEXT PRIMARY KEY,
                    shard_id INTEGER,
                    offset INTEGER
                )
            """)

    def resolve_term(self, term: str) -> Optional[Dict]:
        with sqlite3.connect(self.lex_db_path) as conn:
            cur = conn.execute("SELECT shard_id, offset, vector_id FROM lexemes WHERE term=?", (term,))
            row = cur.fetchone()
            if row:
                return {"shard_id": row[0], "offset": row[1], "vector_id": row[2]}
        return None

    def resolve_vector(self, vector_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.vec_db_path) as conn:
            cur = conn.execute("SELECT shard_id, offset FROM vectors WHERE vector_id=?", (vector_id,))
            row = cur.fetchone()
            if row:
                return {"shard_id": row[0], "offset": row[1]}
        return None
        
    def index_lexeme(self, term: str, shard_id: int, offset: int, vector_id: str):
        with sqlite3.connect(self.lex_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO lexemes (term, shard_id, offset, vector_id) VALUES (?, ?, ?, ?)",
                (term, shard_id, offset, vector_id)
            )

    def index_vector(self, vector_id: str, shard_id: int, offset: int):
        with sqlite3.connect(self.vec_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vectors (vector_id, shard_id, offset) VALUES (?, ?, ?)",
                (vector_id, shard_id, offset)
            )


class LexicalStore:
    """Manages Algebraic Definitions (Source of Truth)"""
    
    def __init__(self, index: Index):
        self.index = index
        self.current_shard_id = 0
        self.current_shard_file = self._get_shard_path(0)
        
    def _get_shard_path(self, shard_id: int) -> Path:
        return SHARDS_DIR / f"shard_{shard_id:04d}.jsonl"
        
    def add_entry(self, definition: AlgebraicDef, vector_id: str) -> None:
        """Append entry to current shard and update index"""
        entry = asdict(definition)
        entry["vector_ref"] = vector_id
        
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        
        # Write to shard
        with open(self.current_shard_file, "a", encoding="utf-8") as f:
            offset = f.tell()
            f.write(line)
            
        # Update Index
        self.index.index_lexeme(definition.term, self.current_shard_id, offset, vector_id)
        
    def get_definition(self, term: str) -> Optional[Dict]:
        loc = self.index.resolve_term(term)
        if not loc:
            return None
            
        path = self._get_shard_path(loc["shard_id"])
        if not path.exists():
            return None
            
        with open(path, "r", encoding="utf-8") as f:
            f.seek(loc["offset"])
            line = f.readline()
            return json.loads(line)


class VectorStore:
    """Manages Calculated Vectors (Result cache)"""
    
    def __init__(self, index: Index):
        self.index = index
        self.current_shard_id = 0
        self.current_shard_file = self._get_shard_path(0)
        
    def _get_shard_path(self, shard_id: int) -> Path:
        return VECTORS_DIR / f"vec_{shard_id:04d}.jsonl"

    def add(self, vector: List[float], term: str, axiom_version: str) -> str:
        """Add vector and return its ID"""
        # Generate deterministic vector ID
        vec_bytes = str(vector).encode('utf-8') + axiom_version.encode('utf-8')
        vid = "sha256:" + hashlib.sha256(vec_bytes).hexdigest()[:16]
        
        # Check integrity
        if len(vector) != 7:
            raise ValueError(f"Vector must be 7D, got {len(vector)}")
            
        entry = {
            "vector_id": vid,
            "term": term,
            "vector": vector,
            "axiom_version": axiom_version,
            "phi_norm": sum(x*x for x in vector)**0.5 # Simple Euclidean norm for check
        }
        
        line = json.dumps(entry) + "\n"
        
        path = self.current_shard_file
        with open(path, "a", encoding="utf-8") as f:
            offset = f.tell()
            f.write(line)
            
        self.index.index_vector(vid, self.current_shard_id, offset)
        return vid

    def get(self, vector_id: str) -> Optional[List[float]]:
        loc = self.index.resolve_vector(vector_id)
        if not loc:
            return None
            
        path = self._get_shard_path(loc["shard_id"])
        with open(path, "r", encoding="utf-8") as f:
            f.seek(loc["offset"])
            line = f.readline()
            data = json.loads(line)
            return data["vector"]


# Initialize Singleton Accessors
_index = Index()
lexicon = LexicalStore(_index)
vectors = VectorStore(_index)

if __name__ == "__main__":
    print("CMFO D9 Storage Test")
    print("====================")
    
    term = "prueba_d9"
    
    # 1. Create Def
    def_obj = AlgebraicDef(
        term=term,
        algebraic_def={"base": ["entidad"], "properties": ["test"]},
        source="manual_test"
    )
    
    # 2. Add Vector
    vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vid = vectors.add(vec, term, "CMFO-D8-v1")
    print(f"Stored Vector ID: {vid}")
    
    # 3. Add Def
    lexicon.add_entry(def_obj, vid)
    print(f"Stored Definition for '{term}'")
    
    # 4. Retrieve
    retrieved_def = lexicon.get_definition(term)
    retrieved_vec = vectors.get(vid)
    
    print(f"Retrieved Def: {retrieved_def['term']}")
    print(f"Retrieved Vec: {retrieved_vec}")
    
    if retrieved_vec == vec:
        print("PASS: Storage Integrity Verified")
    else:
        print("FAIL: Integrity mismatch")
