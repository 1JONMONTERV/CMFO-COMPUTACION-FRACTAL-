
import http.server
import socketserver
import json
import csv
import os
import sys

PORT = 8000
CSV_PATH = "FRACTAL_OMNIVERSE_RECURSIVE.csv"

class FractalHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Read last N lines
            data = {"nodes": [], "links": []}
            nodes_set = set()
            
            try:
                # Efficiently read last 500 lines
                with open(CSV_PATH, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    header = lines[0].strip().split(',')
                    # Take last 500, excluding header if small
                    relevant = lines[-500:] 
                    if relevant[0].startswith("Concept_A"): relevant = relevant[1:]
                    
                    for line in relevant:
                        parts = line.strip().split(',')
                        if len(parts) >= 4:
                            src = parts[0]
                            tgt = parts[1]
                            meaning = parts[2]
                            val = float(parts[3])
                            
                            nodes_set.add(src)
                            nodes_set.add(tgt)
                            
                            data["links"].append({
                                "source": src,
                                "target": tgt,
                                "meaning": meaning,
                                "value": val
                            })
                            
                data["nodes"] = [{"id": n, "group": 1} for n in nodes_set]
                
            except Exception as e:
                print(f"Error reading CSV: {e}")
                
            self.wfile.write(json.dumps(data).encode())
        else:
            # Check if requesting root, serve vis file
            if self.path == '/':
                self.path = '/web/static/fractal_vis.html'
            super().do_GET()

# Ensure we can serve from root to access web/static
# Current dir is root of workspace
print(f"[*] Sirviendo Visualización Fractal en http://localhost:{PORT}")
print(f"    - Visualización: http://localhost:{PORT}/")
print(f"    - Flujo de Datos:   http://localhost:{PORT}/data")

with socketserver.TCPServer(("", PORT), FractalHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
