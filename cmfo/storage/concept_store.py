"""
CMFO D15: Concept Store
=======================
Manages domain-specific databases for Refined Concepts.
Storage: D:/CMFO_DATA/concepts/{domain}.db
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

CONCEPTS_ROOT = Path("D:/CMFO_DATA/concepts")

class ConceptStore:
    def __init__(self, domain: str):
        self.domain = domain
        self.db_path = CONCEPTS_ROOT / f"{domain}.db"
        self._init_db()
        
    def _init_db(self):
        CONCEPTS_ROOT.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        
        # Schema
        c.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                term TEXT PRIMARY KEY,
                definition_formal TEXT,
                vector_json TEXT,
                provenance TEXT,
                concept_type TEXT DEFAULT 'concept',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # FTS for semantic search on definitions
        # FTS Schema Check & Migration
        # SQLite FTS tables don't support ALTER TABLE ADD COLUMN
        try:
            # Check if column exists
            c.execute("SELECT concept_type FROM concepts_fts LIMIT 0")
        except sqlite3.OperationalError:
            # Column missing, need to rebuild FTS
            print("Migrating FTS Schema (Rebuilding Index)...")
            c.execute("DROP TABLE IF EXISTS concepts_fts")
            c.execute("CREATE VIRTUAL TABLE concepts_fts USING fts5(term, definition_formal, concept_type)")
            # Re-populate
            c.execute("INSERT INTO concepts_fts(rowid, term, definition_formal, concept_type) SELECT rowid, term, definition_formal, concept_type FROM concepts")
            conn.commit()
            print("FTS Migration Complete.")

        # Main Table Migration (Safe)
        try:
            c.execute("ALTER TABLE concepts ADD COLUMN concept_type TEXT DEFAULT 'concept'")
        except sqlite3.OperationalError:
            pass # Column already exists
            
        conn.commit()
        conn.close()
        
    def add_concept(self, term: str, definition: str, vector: List[float], source: str, c_type: str = "concept"):
        """Adds or updates a concept"""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        try:
            vec_str = json.dumps(vector)
            c.execute("""
                INSERT OR REPLACE INTO concepts (term, definition_formal, vector_json, provenance, concept_type)
                VALUES (?, ?, ?, ?, ?)
            """, (term, definition, vec_str, source, c_type))
            
            # Update FTS
            c.execute("INSERT OR REPLACE INTO concepts_fts (rowid, term, definition_formal, concept_type) VALUES (last_insert_rowid(), ?, ?, ?)", (term, definition, c_type))
            
            conn.commit()
        except Exception as e:
            print(f"Store Error: {e}")
        finally:
            conn.close()

    def get(self, term: str) -> Optional[Dict]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM concepts WHERE term=?", (term,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
        
    def search(self, query: str) -> List[Dict]:
        """Full text search on definitions"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM concepts_fts WHERE concepts_fts MATCH ? LIMIT 10", (query,))
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def count(self) -> int:
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("SELECT count(*) FROM concepts")
        res = c.fetchone()[0]
        conn.close()
        return res
