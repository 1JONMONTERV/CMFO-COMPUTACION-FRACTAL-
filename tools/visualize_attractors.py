"""
CMFO Attractor Visualizer
=========================
Visualizes the semantic gravity wells (Attractors) from the Fractal Omniverse.
"""

import csv
import os
import sys
import random
import math

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("WARNING: matplotlib or networkx not found. Running in Text Mode.")

def load_data(filepath):
    edges = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dist = float(row['Resonance_Distance'])
                # Filter: Only visualize strong connections (Resonance < 1.1)
                # or very meaningful ones.
                if dist < 1.2: 
                    edges.append({
                        'source': row['Concept_A'],
                        'target': row['Concept_B'],
                        'weight': 1.0 / (dist + 1e-6), # Inverse distance = Gravity
                        'meaning': row['Emergent_Meaning']
                    })
            except ValueError:
                pass
    return edges

def visualize(edges):
    if not HAS_VIZ:
        print(f"Loaded {len(edges)} resonant connections.")
        print("Top 10 Strongest Connections:")
        sorted_edges = sorted(edges, key=lambda x: x['weight'], reverse=True)
        for e in sorted_edges[:10]:
            print(f"  {e['source']} <==> {e['target']} (Strength: {e['weight']:.2f}) -> {e['meaning']}")
        return

    G = nx.Graph()
    
    print(f"Building Graph with {len(edges)} strong connections...")
    
    for e in edges:
        G.add_edge(e['source'], e['target'], weight=e['weight'])

    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    plt.figure(figsize=(12, 12), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='#FFD700', alpha=0.8)

    # Draw Edges (faint)
    nx.draw_networkx_edges(G, pos, edge_color='#444444', alpha=0.3)

    # Labels (only for high degrees)
    degrees = dict(G.degree)
    important_nodes = {n for n, d in degrees.items() if d > 2}
    labels = {n: n for n in important_nodes}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='white', font_family='sans-serif')

    plt.title("CMFO Fractal Attractor Map (Resonance < 1.2)", color='white', fontsize=16)
    plt.axis('off')
    
    output_path = "attractor_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {os.path.abspath(output_path)}")
    # plt.show() # Uncomment to show window

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Assuming script is in tools/ or root
    # Try finding the CSV
    csv_path = os.path.join(base_dir, "FRACTAL_OMNIVERSE_RECURSIVE.csv")
    
    if not os.path.exists(csv_path):
        # Fallback to current dir
        csv_path = "FRACTAL_OMNIVERSE_RECURSIVE.csv"
    
    if os.path.exists(csv_path):
        edges = load_data(csv_path)
        visualize(edges)
    else:
        print(f"Error: Could not find FRACTAL_OMNIVERSE_RECURSIVE.csv")
