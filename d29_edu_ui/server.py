import http.server
import socketserver
import json
import os
import sys

# Bridge to Core
sys.path.append(os.getcwd())
# Ensure we map imports correctly if running as script from root
# If this script is run from root (python d29_edu_ui/server.py), os.getcwd is root.

try:
    from d27_edu_core.secure_tutor import SecureTutor
except ImportError:
    # Fallback if path issues
    print("Error importing SecureTutor. Check python path.")
    sys.exit(1)

# Initialize Global Tutor
# Ideally this is per-session, but for single-user prototype global is fine
TUTOR = SecureTutor("d26_edu_pilot/syllabus_math_10.json", tutor_identity="gerente")

class CMFORequestHeader(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Redirect root to our interface
        if self.path == '/' or self.path == '/index.html':
            self.path = '/d29_edu_ui/index.html'
        
        # Disable cache for dev
        self.send_response(200)
        
        # Infer MIME
        if self.path.endswith(".html"):
            self.send_header('Content-type', 'text/html')
        elif self.path.endswith(".css"):
            self.send_header('Content-type', 'text/css')
        elif self.path.endswith(".js"):
            self.send_header('Content-type', 'application/javascript')
        else:
             # Let superclass handle or default
             return http.server.SimpleHTTPRequestHandler.do_GET(self)
             
        self.end_headers()
        
        # Serve file
        # We need absolute path relative to CWD
        try:
            # removing leading / 
            clean_path = self.path.lstrip('/')
            with open(clean_path, 'rb') as f:
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.send_error(404, f"File not found: {self.path}")

    def do_POST(self):
        if self.path == '/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                user_query = data.get('query')
                
                # CMFO INTERACTION
                print(f"[UI-SERVER] Query: {user_query}")
                result = TUTOR.interact(user_query)
                
                # Structure Debug
                # SecureTutor returns: {"response": {"status":..., "response":...}, "receipt": ...}
                
                tutor_resp = result["response"]
                receipt = result["receipt"]
                
                # Check structure validity (audit lock)
                # SecureTutor logic already normalizes 'ERROR' to 'BLOCKED' msg usually
                
                response_payload = {
                    "tutor_text": tutor_resp.get("response", ""),
                    "status": tutor_resp.get("status", "UNKNOWN"), # AUTHORIZED, BLOCKED, AXIOM_VIOLATION
                    "violation": tutor_resp.get("violation"),
                    "receipt_hash": receipt.get("ciphertext", "N/A")[:24] + "..." # Visual Proof
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
                
            except Exception as e:
                print(f"Error processing chat: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404)

PORT = 8088

def run_server():
    # Allow serving from root to access d29_edu_ui
    # We must run this from project root
    print(f"[*] CMFO Sovereign Interface Active at http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), CMFORequestHeader) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            httpd.server_close()

if __name__ == "__main__":
    run_server()
