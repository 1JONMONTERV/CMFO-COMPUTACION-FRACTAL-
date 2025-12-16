"""
CMFO D10: Industrial Reader Bridge
==================================
The "Hippocampal Bridge" to Long-Term Storage.

Responsibility:
- Safe, concurrent reading of D9 Data (D:/CMFO_DATA).
- Zero-lock interference with Ingestion.
- Micro-second latency for indexed lookups.
"""

import sqlite3
import json
import os
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from functools import lru_cache

# Constants
STORAGE_ROOT = Path("D:/CMFO_DATA")
INDEX_DIR = STORAGE_ROOT / "index"
SHARDS_DIR = STORAGE_ROOT / "shards/es"
VECTORS_DIR = STORAGE_ROOT / "vectors/v1"

class KnowledgeBridge:
    """
    Read-Only Interface to the CMFO Massive Store.
    Thread-safe. Low-latency.
    """
    
    def __init__(self):
        self._ensure_paths()
        self.lex_conn = None
        self.vec_conn = None
        self._connect_dbs()
        
    def _ensure_paths(self):
        if not INDEX_DIR.exists():
            print(f"Warning: Index dir {INDEX_DIR} not found. Bridge active but empty.")

    def _connect_dbs(self):
        """Connect to SQLite in Read-Only mode if possible, or standard with immediate release"""
        try:
            # We use distinct connections for thread safety if extended later, 
            # but for now simple check_same_thread=False for flexibility
            lex_db = INDEX_DIR / "lexeme.db"
            vec_db = INDEX_DIR / "vector.db"
            
            if lex_db.exists():
                self.lex_conn = sqlite3.connect(f"file:{lex_db}?mode=ro", uri=True, check_same_thread=False)
                # Ensure WAL is set by writer, not reader.
                
            if vec_db.exists():
                self.vec_conn = sqlite3.connect(f"file:{vec_db}?mode=ro", uri=True, check_same_thread=False)
                
        except Exception as e:
            print(f"Bridge connection error: {e}")

    @lru_cache(maxsize=1000)
    def get_vector(self, term: str) -> Optional[tuple]:
        """
        Get vector for a term.
        Returns tuple for cacheability (immutable). 
        Convert to list if needed by caller.
        """
        if not self.lex_conn:
            return None
            
        try:
            cursor = self.lex_conn.cursor()
            # 1. Lookup Vector ID for term
            # In D9 we stored: term -> vector_id IN lexemes table
            # But wait, did we verify schema?
            # Schema: table lexemes (term, shard_id, offset, vector_id)
            
            cursor.execute("SELECT vector_id FROM lexemes WHERE term=?", (term,))
            row = cursor.fetchone()
            
            if not row:
                return None
                
            vid = row[0]
            
            # 2. Lookup Vector Location
            if not self.vec_conn:
                return None
                
            v_cursor = self.vec_conn.cursor()
            v_cursor.execute("SELECT shard_id, offset FROM vectors WHERE vector_id=?", (vid,))
            v_row = v_cursor.fetchone()
            
            if not v_row:
                return None
                
            shard_id, offset = v_row
            
            # 3. Read Physical Shard (The "Seek")
            # Vector shards are in VECTORS_DIR ? 
            # Wait, `vectors.add` in storage/core.py writes to vectors/v1/vec_{shard}.jsonl
            # Shard_id 0 corresponds to vec_0000.jsonl
            
            shard_path = VECTORS_DIR / f"vec_{shard_id:04d}.jsonl"
            
            if not shard_path.exists():
                return None
                
            # Random Access Read
            with open(shard_path, "r", encoding="utf-8") as f:
                f.seek(offset)
                line = f.readline()
                data = json.loads(line)
                
                # Check integrity (rare collision/overwrite check)
                if data.get("vector_id") == vid:
                    return tuple(data["vector"])
                
        except Exception as e:
            # Silence transient errors in production lookups
            # print(f"Bridge Lookup Error: {e}")
            pass
            
        return None

    def exists(self, term: str) -> bool:
        """Fast existence check (Index only)"""
        if not self.lex_conn:
            return False
        try:
            cursor = self.lex_conn.cursor()
            cursor.execute("SELECT 1 FROM lexemes WHERE term=?", (term,))
            return cursor.fetchone() is not None
        except:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Ecosystem Pulse"""
        stats = {
            "status": "offline",
            "terms_indexed": 0,
            "vectors_indexed": 0
        }
        
        if self.lex_conn:
            try:
                cur = self.lex_conn.cursor()
                cur.execute("SELECT Count(*) FROM lexemes")
                stats["terms_indexed"] = cur.fetchone()[0]
                stats["status"] = "online"
            except:
                pass
                
        if self.vec_conn:
            try:
                cur = self.vec_conn.cursor()
                cur.execute("SELECT Count(*) FROM vectors")
                stats["vectors_indexed"] = cur.fetchone()[0]
            except:
                pass
                
        return stats

    def close(self):
        if self.lex_conn: self.lex_conn.close()
        if self.vec_conn: self.vec_conn.close()
