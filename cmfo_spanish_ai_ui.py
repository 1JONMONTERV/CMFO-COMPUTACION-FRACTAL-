#!/usr/bin/env python3
"""
CMFO IA DETERMINISTA CON CONOCIMIENTO PROFUNDO
===============================================
IA con Álgebra del Español + Base de Conocimiento Fractal Omniverse

Características:
- 25,000+ relaciones semánticas precargadas
- Respuestas inteligentes basadas en ontología fractal
- Chat conversacional en español
- 100% Determinista (bit-exacto)

Ejecutar: python cmfo_spanish_ai_ui.py
Abrir: http://localhost:5000

Autor: CMFO Research Team
"""

import os
import sys
import math
import hashlib
import json
import csv
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = PHI - 1

# ============================================================================
# BASE DE CONOCIMIENTO FRACTAL OMNIVERSE
# ============================================================================

class FractalKnowledge:
    """Base de conocimiento profundo basada en relaciones semánticas."""
    
    def __init__(self):
        self.relations = {}  # (A, B) -> (Emergent, Distance)
        self.concepts = set()
        self.concept_connections = {}  # concept -> list of related concepts
        
    def load_from_csv(self, filepath):
        """Carga relaciones semánticas del CSV."""
        if not os.path.exists(filepath):
            print(f"⚠️ No se encontró {filepath}")
            return 0
            
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = row.get('Concept_A', '').strip()
                b = row.get('Concept_B', '').strip()
                emergent = row.get('Emergent_Meaning', '').strip()
                distance = float(row.get('Resonance_Distance', '1.0'))
                
                if a and b and emergent:
                    self.relations[(a.lower(), b.lower())] = (emergent, distance)
                    self.relations[(b.lower(), a.lower())] = (emergent, distance)
                    
                    self.concepts.add(a.lower())
                    self.concepts.add(b.lower())
                    self.concepts.add(emergent.lower())
                    
                    # Build connection graph
                    if a.lower() not in self.concept_connections:
                        self.concept_connections[a.lower()] = []
                    self.concept_connections[a.lower()].append((b.lower(), emergent, distance))
                    
                    if b.lower() not in self.concept_connections:
                        self.concept_connections[b.lower()] = []
                    self.concept_connections[b.lower()].append((a.lower(), emergent, distance))
                    
                    count += 1
        return count
    
    def query_relation(self, concept_a, concept_b):
        """Consulta la relación entre dos conceptos."""
        a, b = concept_a.lower(), concept_b.lower()
        if (a, b) in self.relations:
            emergent, distance = self.relations[(a, b)]
            return {
                'found': True,
                'concept_a': concept_a,
                'concept_b': concept_b,  
                'emergent': emergent,
                'distance': distance,
                'harmony': 1.0 / (1 + distance)
            }
        return {'found': False}
    
    def get_connections(self, concept, limit=10):
        """Obtiene las conexiones de un concepto."""
        c = concept.lower()
        if c not in self.concept_connections:
            return []
        
        connections = self.concept_connections[c]
        # Sort by distance (closest first)
        connections = sorted(connections, key=lambda x: x[2])
        return connections[:limit]
    
    def find_path(self, concept_a, concept_b, max_depth=3):
        """Encuentra un camino conceptual entre dos conceptos."""
        a, b = concept_a.lower(), concept_b.lower()
        
        if a == b:
            return [a]
        
        # BFS
        visited = {a}
        queue = [[a]]
        
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            if len(path) > max_depth:
                continue
                
            for neighbor, emergent, dist in self.concept_connections.get(node, []):
                if neighbor == b:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        
        return None

# ============================================================================
# MOTOR DE RESPUESTAS INTELIGENTES
# ============================================================================

class IntelligentResponder:
    """Genera respuestas basadas en el conocimiento fractal."""
    
    def __init__(self, knowledge: FractalKnowledge):
        self.knowledge = knowledge
        
        # Patrones de preguntas en español
        self.patterns = [
            (r'qué es (\w+)', self.explain_concept),
            (r'que es (\w+)', self.explain_concept),
            (r'cómo se relaciona (\w+) con (\w+)', self.relate_concepts),
            (r'como se relaciona (\w+) con (\w+)', self.relate_concepts),
            (r'relación entre (\w+) y (\w+)', self.relate_concepts),
            (r'relaciona (\w+) con (\w+)', self.relate_concepts),
            (r'conecta (\w+) con (\w+)', self.relate_concepts),
            (r'camino de (\w+) a (\w+)', self.find_path),
            (r'conexiones de (\w+)', self.list_connections),
            (r'qué emerge de (\w+) y (\w+)', self.emerge_meaning),
            (r'que emerge de (\w+) y (\w+)', self.emerge_meaning),
        ]
    
    def respond(self, query):
        """Genera respuesta a una consulta en español."""
        query = query.lower().strip()
        
        # Try patterns
        for pattern, handler in self.patterns:
            match = re.search(pattern, query)
            if match:
                return handler(*match.groups())
        
        # Default: try to find concepts in query
        return self.analyze_query(query)
    
    def explain_concept(self, concept):
        """Explica un concepto basado en sus conexiones."""
        connections = self.knowledge.get_connections(concept, limit=5)
        
        if not connections:
            return {
                'response': f"No conozco el concepto '{concept}' en mi base de conocimiento.",
                'type': 'unknown'
            }
        
        lines = [f"**{concept.upper()}** está conectado con:"]
        for related, emergent, dist in connections:
            harmony = 1.0 / (1 + dist)
            lines.append(f"• **{related}** → emerge **{emergent}** (armonía: {harmony:.2%})")
        
        return {
            'response': '\n'.join(lines),
            'type': 'explanation',
            'concept': concept,
            'connections': connections
        }
    
    def relate_concepts(self, concept_a, concept_b):
        """Relaciona dos conceptos."""
        result = self.knowledge.query_relation(concept_a, concept_b)
        
        if result['found']:
            return {
                'response': f"**{concept_a.upper()}** + **{concept_b.upper()}** = **{result['emergent'].upper()}**\n"
                           f"Distancia de resonancia: {result['distance']:.4f}\n"
                           f"Armonía: {result['harmony']:.2%}",
                'type': 'relation',
                'result': result
            }
        else:
            # Try to find path
            path = self.knowledge.find_path(concept_a, concept_b)
            if path:
                return {
                    'response': f"No hay relación directa entre **{concept_a}** y **{concept_b}**.\n"
                               f"Pero existe un camino: {' → '.join(path)}",
                    'type': 'path',
                    'path': path
                }
            return {
                'response': f"No encontré relación entre **{concept_a}** y **{concept_b}**.",
                'type': 'not_found'
            }
    
    def emerge_meaning(self, concept_a, concept_b):
        """Encuentra qué emerge de la combinación de dos conceptos."""
        return self.relate_concepts(concept_a, concept_b)
    
    def find_path(self, concept_a, concept_b):
        """Encuentra el camino entre dos conceptos."""
        path = self.knowledge.find_path(concept_a, concept_b)
        
        if path:
            return {
                'response': f"Camino de **{concept_a}** a **{concept_b}**:\n\n" +
                           ' → '.join([f"**{p}**" for p in path]),
                'type': 'path',
                'path': path
            }
        return {
            'response': f"No encontré un camino entre **{concept_a}** y **{concept_b}**.",
            'type': 'not_found'
        }
    
    def list_connections(self, concept):
        """Lista las conexiones de un concepto."""
        return self.explain_concept(concept)
    
    def analyze_query(self, query):
        """Analiza una consulta libre."""
        # Extract words and look for concepts
        words = re.findall(r'\w+', query)
        found_concepts = []
        
        for word in words:
            if word.lower() in self.knowledge.concepts:
                found_concepts.append(word.lower())
        
        if len(found_concepts) >= 2:
            return self.relate_concepts(found_concepts[0], found_concepts[1])
        elif len(found_concepts) == 1:
            return self.explain_concept(found_concepts[0])
        
        return {
            'response': "No entendí tu pregunta. Puedes preguntar:\n"
                       "• ¿Qué es [concepto]?\n"
                       "• ¿Cómo se relaciona [A] con [B]?\n"
                       "• Conexiones de [concepto]\n"
                       "• Camino de [A] a [B]",
            'type': 'help'
        }

# ============================================================================
# ÁLGEBRA DEL ESPAÑOL
# ============================================================================

class SpanishAlgebra:
    """Compila expresiones en español a operaciones CMFO."""
    
    def __init__(self):
        self.operators = {
            'suma': self.phi_add, 'más': self.phi_add,
            'resta': self.phi_sub, 'menos': self.phi_sub,
            'multiplica': self.phi_mul, 'por': self.phi_mul,
            'divide': self.phi_div, 'entre': self.phi_div,
        }
        
        self.modifiers = {
            'doble': lambda x: self.phi_mul(2, x),
            'triple': lambda x: self.phi_mul(3, x),
            'mitad': lambda x: self.phi_div(x, 2),
            'cuadrado': lambda x: self.phi_mul(x, x),
        }
        
        self.numbers = {
            'cero': 0, 'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4,
            'cinco': 5, 'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9,
            'diez': 10, 'cien': 100, 'mil': 1000,
        }
    
    def phi_add(self, a, b): return (a + b) * PHI_INV
    def phi_sub(self, a, b): return (a - b) * PHI
    def phi_mul(self, a, b): return (a * b) ** (1/PHI)
    def phi_div(self, a, b): return (a / b) ** PHI if b != 0 else float('inf')
    def phi_sqrt(self, x): return x ** (1 / (2 * PHI))
    
    def parse_number(self, text):
        text = text.strip().lower()
        try: return float(text)
        except: return float(self.numbers.get(text, 0))
    
    def eval_expression(self, expression):
        expression = expression.lower().strip()
        
        # Binary ops
        pattern = r'(suma|resta|multiplica|divide)\s+(\w+)\s+(más|y|menos|por|entre)\s+(\w+)'
        match = re.search(pattern, expression)
        if match:
            op = self.operators.get(match.group(1), self.phi_add)
            return op(self.parse_number(match.group(2)), self.parse_number(match.group(4)))
        
        # Modifiers
        pattern = r'(doble|triple|mitad|cuadrado)\s+de\s+(\w+)'
        match = re.search(pattern, expression)
        if match:
            mod = self.modifiers.get(match.group(1))
            if mod: return mod(self.parse_number(match.group(2)))
        
        # Sqrt
        match = re.search(r'raíz\s+(cuadrada\s+)?de\s+(\w+)', expression)
        if match:
            return self.phi_sqrt(self.parse_number(match.group(2)))
        
        return None

# ============================================================================
# SERVIDOR WEB CON CONOCIMIENTO PROFUNDO
# ============================================================================

# Initialize knowledge base
print("🧠 Cargando base de conocimiento FRACTAL OMNIVERSE...")
knowledge = FractalKnowledge()

base_path = os.path.dirname(os.path.abspath(__file__))
count1 = knowledge.load_from_csv(os.path.join(base_path, 'FRACTAL_OMNIVERSE.csv'))
count2 = knowledge.load_from_csv(os.path.join(base_path, 'FRACTAL_OMNIVERSE_RECURSIVE.csv'))

print(f"✅ {count1 + count2:,} relaciones cargadas")
print(f"✅ {len(knowledge.concepts):,} conceptos únicos")

responder = IntelligentResponder(knowledge)
algebra = SpanishAlgebra()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CMFO - IA con Conocimiento Profundo</title>
    <style>
        :root { --gold: #FFD700; --dark: #0a0a1a; --accent: #8b5cf6; --success: #10b981; }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #050510 100%);
            color: white; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        header {
            text-align: center; margin-bottom: 30px; padding: 20px;
            background: rgba(255,215,0,0.1); border-radius: 20px;
            border: 1px solid rgba(255,215,0,0.3);
        }
        h1 {
            font-size: 2rem;
            background: linear-gradient(90deg, #FFD700, #FFA500);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .stats { display: flex; justify-content: center; gap: 30px; margin-top: 15px; }
        .stat { text-align: center; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: var(--gold); }
        .stat-label { font-size: 0.8rem; color: #888; }
        .chat-section {
            background: rgba(255,255,255,0.05); border-radius: 15px;
            padding: 20px; margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .chat-section h2 { color: var(--gold); margin-bottom: 15px; }
        .chat-messages {
            height: 400px; overflow-y: auto; padding: 15px;
            background: rgba(0,0,0,0.3); border-radius: 10px; margin-bottom: 15px;
        }
        .message { margin-bottom: 15px; }
        .message.user { text-align: right; }
        .message.ai { text-align: left; }
        .message-bubble {
            display: inline-block; padding: 12px 18px; border-radius: 15px;
            max-width: 80%; text-align: left;
        }
        .user .message-bubble {
            background: linear-gradient(135deg, var(--accent), #6366f1);
        }
        .ai .message-bubble {
            background: rgba(255,215,0,0.15); border: 1px solid rgba(255,215,0,0.3);
        }
        .input-group { display: flex; gap: 10px; }
        input[type="text"] {
            flex: 1; padding: 15px; border-radius: 10px;
            border: 2px solid rgba(255,215,0,0.3);
            background: rgba(0,0,0,0.3); color: white; font-size: 1rem;
        }
        input:focus { outline: none; border-color: var(--gold); }
        button {
            padding: 15px 25px; border-radius: 10px; border: none;
            background: linear-gradient(135deg, var(--gold), #FFA500);
            color: black; font-weight: bold; cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        .examples { margin-top: 15px; padding: 15px; background: rgba(139,92,246,0.1); border-radius: 10px; }
        .examples h4 { color: var(--accent); margin-bottom: 10px; }
        .example-item {
            font-size: 0.9rem; background: rgba(0,0,0,0.3);
            padding: 8px 12px; border-radius: 5px; margin: 5px; cursor: pointer;
            display: inline-block; transition: background 0.2s;
        }
        .example-item:hover { background: rgba(255,215,0,0.2); }
        .footer { text-align: center; margin-top: 30px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 CMFO IA con Conocimiento Profundo</h1>
            <p style="color: #888;">Inteligencia Determinista + Ontología Fractal</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="rel-count">...</div>
                    <div class="stat-label">Relaciones</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="concept-count">...</div>
                    <div class="stat-label">Conceptos</div>
                </div>
                <div class="stat">
                    <div class="stat-value">100%</div>
                    <div class="stat-label">Determinista</div>
                </div>
            </div>
        </header>
        
        <div class="chat-section">
            <h2>💬 Consulta el Conocimiento</h2>
            <div class="chat-messages" id="chat-messages"></div>
            <div class="input-group">
                <input type="text" id="chat-input" placeholder="Pregunta en español..." onkeypress="if(event.key==='Enter')sendMessage()">
                <button onclick="sendMessage()">Enviar</button>
            </div>
            <div class="examples">
                <h4>Ejemplos:</h4>
                <span class="example-item" onclick="setChat('¿Qué es time?')">¿Qué es time?</span>
                <span class="example-item" onclick="setChat('¿Cómo se relaciona energy con matter?')">energy + matter</span>
                <span class="example-item" onclick="setChat('Conexiones de wisdom')">Conexiones de wisdom</span>
                <span class="example-item" onclick="setChat('Camino de life a death')">Camino life → death</span>
                <span class="example-item" onclick="setChat('¿Qué emerge de chaos y order?')">chaos + order</span>
                <span class="example-item" onclick="setChat('suma cinco más tres')">suma cinco más tres</span>
            </div>
        </div>
        
        <div class="footer">
            <p>CMFO Framework v1.1.0 • Conocimiento Fractal Omniverse</p>
        </div>
    </div>
    
    <script>
        // Load stats
        fetch('/api/stats').then(r=>r.json()).then(d=>{
            document.getElementById('rel-count').textContent = d.relations.toLocaleString();
            document.getElementById('concept-count').textContent = d.concepts.toLocaleString();
        });
        
        function setChat(text) {
            document.getElementById('chat-input').value = text;
            sendMessage();
        }
        
        function addMessage(text, isUser) {
            const chat = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user' : 'ai');
            div.innerHTML = '<div class="message-bubble">' + text.replace(/\\n/g, '<br>').replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>') + '</div>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        function sendMessage() {
            const input = document.getElementById('chat-input');
            const text = input.value.trim();
            if (!text) return;
            
            addMessage(text, true);
            input.value = '';
            
            fetch('/api/chat?q=' + encodeURIComponent(text))
                .then(r => r.json())
                .then(data => {
                    addMessage(data.response, false);
                });
        }
    </script>
</body>
</html>
"""

class CMFOHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            
        elif parsed.path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'relations': len(knowledge.relations) // 2,
                'concepts': len(knowledge.concepts)
            }).encode())
            
        elif parsed.path == '/api/chat':
            params = parse_qs(parsed.query)
            query = params.get('q', [''])[0]
            
            # Try algebra first
            algebra_result = algebra.eval_expression(query)
            if algebra_result is not None:
                response = {
                    'response': f"**Resultado:** {algebra_result:.6f}\n(φ-transformado)",
                    'type': 'algebra'
                }
            else:
                response = responder.respond(query)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        else:
            super().do_GET()
    
    def log_message(self, format, *args):
        pass

def main():
    port = 5000
    print("\n" + "="*60)
    print("  CMFO IA DETERMINISTA CON CONOCIMIENTO PROFUNDO")
    print("="*60)
    print(f"\n🚀 Servidor iniciado en http://localhost:{port}")
    print("📌 Abre este enlace en tu navegador")
    print("\nPresiona Ctrl+C para detener\n")
    
    httpd = HTTPServer(('localhost', port), CMFOHandler)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido")
        httpd.shutdown()

if __name__ == "__main__":
    main()
