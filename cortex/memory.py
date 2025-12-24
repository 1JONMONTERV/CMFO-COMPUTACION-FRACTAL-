
import sys
import heapq
sys.path.insert(0, '../')
from cortex.encoder import FractalEncoder

class FractalMemory:
    """
    The Deterministic Knowledge Graph.
    Stores concepts as points in the 7D Manifold.
    Retrieves them by Geometric Resonance (Distance).
    """
    def __init__(self):
        self.encoder = FractalEncoder()
        self.vectors = {} # map: vector_tuple -> data
        self.keys = []
        
    def learn(self, concept_name, definition):
        """
        Indexes a piece of knowledge.
        The 'Key' is the semantic vector of the concept name.
        """
        vec = self.encoder.encode(concept_name)
        # Store using tuple as dict key (hashable)
        k = tuple(vec.v)
        self.vectors[k] = {
            'name': concept_name,
            'def': definition,
            'vec_obj': vec
        }
        self.keys.append(k)
        # print(f"[*] Learned: {concept_name}") # Silent learning
        
    def query(self, query_text, top_k=3):
        """
        Finds the most relevant concepts deterministically.
        """
        q_vec = self.encoder.encode(query_text)
        
        # Linear Scan (perfect accuracy for prototype)
        # In production v4.0, use Fractal-KD-Tree
        distances = []
        for k in self.keys:
            stored_data = self.vectors[k]
            # Use Encoder's metric
            dist = self.encoder.conceptual_distance(q_vec, stored_data['vec_obj'])
            heapq.heappush(distances, (dist, stored_data))
            
        # Retrieve top K
        results = []
        for _ in range(min(top_k, len(distances))):
            dist, data = heapq.heappop(distances)
            results.append((data, dist))
            
        return results

if __name__ == "__main__":
    mem = FractalMemory()
    mem.learn("Energy", "The capacity for doing work.")
    mem.learn("Mass", "A property of a physical body and a measure of its resistance to acceleration.")
    mem.learn("Speed of Light", "The speed limit of the universe.")
    
    print("Memory Initialized.")
    res = mem.query("Power", top_k=1)
    print(f"Query 'Power' matched: {res[0][0]['name']} (Dist: {res[0][1]:.4f})")
