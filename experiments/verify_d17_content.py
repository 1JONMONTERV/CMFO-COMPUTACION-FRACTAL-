"""
CMFO D17: Verification
======================
Verifies the existence of Structural Concepts (Theorems, Proofs)
in the populated database.
"""
import sqlite3
import sys

# Force UTF-8 Output
sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = "D:/CMFO_DATA/concepts/computacion.db"

def verify():
    print(f"Connecting to {DB_PATH}...")
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # 1. Count by Type
        print("\n[STATS]")
        c.execute("SELECT concept_type, COUNT(*) FROM concepts GROUP BY concept_type")
        for row in c.fetchall():
            print(f"  {row[0]}: {row[1]}")
            
        # 2. Show Theorems
        print("\n[SAMPLE THEOREMS]")
        c.execute("SELECT term, definition_formal FROM concepts_fts WHERE concept_type='theorem' LIMIT 3")
        for row in c.fetchall():
            print(f"  * {row[0]}: {row[1][:100]}...")
            
        # 3. Show Proofs
        print("\n[SAMPLE PROOFS]")
        c.execute("SELECT term, definition_formal FROM concepts_fts WHERE concept_type='proof' LIMIT 1")
        for row in c.fetchall():
            print(f"  * {row[0]}: {row[1][:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    verify()
