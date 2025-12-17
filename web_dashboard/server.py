"""
CMFO Fractal Dashboard Backend
==============================
Serves the Omniverse to the Web.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import csv
import os
import sys
import random
import time

# Add root and bindings to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'bindings', 'python'))

try:
    import cmfo
    from cmfo.constants import PHI
    # Try importing the JIT bridge indirectly via core
    from cmfo.core.gamma_phi import gamma_step
except ImportError:
    print("WARNING: CMFO package not found. Running in mock mode.")
    cmfo = None

app = FastAPI(title="CMFO Fractal Dashboard")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FRACTAL_OMNIVERSE_RECURSIVE.csv")

class DreamRequest(BaseModel):
    concept_a: str
    concept_b: str

@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/api/graph")
async def get_graph():
    """Reads the CSV and returns a 3D graph structure."""
    nodes = {} # Set of unique node names
    links = []
    
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = row['Concept_A']
                target = row['Concept_B']
                meaning = row['Emergent_Meaning']
                try:
                    dist = float(row['Resonance_Distance'])
                except:
                    dist = 2.0 # Default weak
                
                # Add nodes if new
                if source not in nodes: nodes[source] = {"id": source, "group": 1}
                if target not in nodes: nodes[target] = {"id": target, "group": 1}
                if meaning not in nodes: nodes[meaning] = {"id": meaning, "group": 2} # Emerging concepts are group 2
                
                # Links: Parents -> Child
                # Invert logic: The relationship creates the child.
                # Visualization: Source & Target connect to Meaning
                links.append({"source": source, "target": meaning, "val": 1.0/dist})
                links.append({"source": target, "target": meaning, "val": 1.0/dist})

    return {
        "nodes": list(nodes.values()),
        "links": links
    }

@app.post("/api/dream")
async def trigger_dream():
    """Triggers a single JIT Dream cycle and appends to CSV."""
    if not cmfo:
        return {"status": "error", "message": "CMFO Core not loaded"}

    # 1. Load available concepts
    concepts = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                concepts.append(row['Concept_A'])
                concepts.append(row['Concept_B'])
    concepts = list(set(concepts))
    
    if not concepts:
         concepts = ["Void", "Light", "Time"]

    # 2. Pick Parents
    p1 = random.choice(concepts)
    p2 = random.choice(concepts)
    
    # 3. Execute Manifold (JIT inside)
    manifold = cmfo.PhiManifold(7)
    result = manifold.dream(p1, p2)
    
    # 4. Save
    with open(CSV_PATH, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([p1, p2, result.meaning, f"{result.resonance:.4f}"])
        
    return {
        "status": "success",
        "dream": {
            "parents": [p1, p2],
            "result": result.meaning,
            "resonance": result.resonance
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
