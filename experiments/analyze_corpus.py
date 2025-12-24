"""
CMFO D9: Corpus Analysis (What did we ingest?)
==============================================
Reads the SQLite Index and random vector shards to provide a
detailed report on the contents of the D: Drive.
"""

import sqlite3
import random
import sys
from pathlib import Path

INDEX_PATH = "D:/CMFO_DATA/index/lexeme.db"

def analyze_corpus():
    print("ANALYZING CMFO CORPUS (D:)...")
    
    if not Path(INDEX_PATH).exists():
        print("Error: Index not found.")
        return

    conn = sqlite3.connect(f"file:{INDEX_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    
    # 1. Total Count
    cur.execute("SELECT Count(*) FROM lexemes")
    total = cur.fetchone()[0]
    print(f"\n[1] TOTAL VOLUME: {total:,} Defined Concepts")
    
    if total == 0:
        return
        
    # 2. Sample Terms (Random)
    print("\n[2] RANDOM SAMPLES (Semantic Diversity Check):")
    # SQLite random is slow on large tables, strict offset is better or just iterate a bit
    # We use random rowid approximation
    
    samples = []
    for _ in range(5):
        rid = random.randint(1, total)
        cur.execute("SELECT term FROM lexemes WHERE rowid=?", (rid,))
        row = cur.fetchone()
        if row:
            samples.append(row[0])
            
    for s in samples:
        print(f"    - {s}")
        
    # 3. Specific Categories Check (Heuristic)
    print("\n[3] CATEGORY COVERAGE (Heuristic Check):")
    
    categories = {
        "Science": ["átomo", "molécula", "gravedad", "electrón"],
        "Biology": ["célula", "mitocondria", "tigre", "bacteria"],
        "Computing": ["algoritmo", "byte", "servidor"],
        "Abstract": ["libertad", "tiempo", "democracia"]
    }
    
    for cat, terms in categories.items():
        found = []
        for t in terms:
            cur.execute("SELECT 1 FROM lexemes WHERE term=?", (t,))
            if cur.fetchone():
                found.append(t)
        
        percent = (len(found) / len(terms)) * 100
        print(f"    - {cat}: {percent:.0f}% ({', '.join(found)})")
        
    print("\n[4] DATA INTEGRITY")
    print(f"    Index Database Size: {Path(INDEX_PATH).stat().st_size / 1024 / 1024:.2f} MB")
    
    conn.close()

if __name__ == "__main__":
    analyze_corpus()
