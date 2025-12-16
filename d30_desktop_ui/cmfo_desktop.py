import tkinter as tk
from tkinter import ttk, scrolledtext
import sys
import os
from datetime import datetime
import threading
import time
import queue

# Add project root to path
sys.path.append(os.getcwd())

try:
    from d27_edu_core.secure_tutor import SecureTutor
except ImportError:
    print("Error: No se pudo importar SecureTutor. Asegúrate de estar en la raíz del proyecto.")
    sys.exit(1)

class ModernScrollableFrame(tk.Frame):
    """Custom scrollable frame for chat messages"""
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, bg='#ffffff', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#ffffff')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
    def scroll_to_bottom(self):
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

class CMFODesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CMFO Tutor Soberano")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f7f7f8')
        
        # Initialize tutor
        print("[*] Inicializando CMFO Tutor...")
        self.tutor = SecureTutor("d26_edu_pilot/syllabus_math_10.json", tutor_identity="gerente")
        print("[*] Tutor listo.")
        
        # Queue for thread-safe communication
        self.response_queue = queue.Queue()
        
        # Colors (ChatGPT style)
        self.colors = {
            'bg_main': '#ffffff',
            'bg_sidebar': '#f7f7f8',
            'bg_user': '#f4f4f4',
            'bg_assistant': '#ffffff',
            'accent': '#10a37f',
            'text_primary': '#202123',
            'text_secondary': '#8e8ea0',
            'border': '#e5e5e5'
        }
        
        self.setup_ui()
        self.show_welcome()
        
    def setup_ui(self):
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg_main'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar
        self.create_sidebar(main_container)
        
        # Chat area
        self.create_chat_area(main_container)
        
    def create_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=self.colors['bg_sidebar'], width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)
        
        # Logo
        logo_frame = tk.Frame(sidebar, bg=self.colors['bg_sidebar'])
        logo_frame.pack(pady=20, padx=15)
        
        logo_label = tk.Label(
            logo_frame, 
            text="CMFO", 
            font=("Inter", 18, "bold"),
            bg=self.colors['bg_sidebar'],
            fg=self.colors['text_primary']
        )
        logo_label.pack()
        
        subtitle = tk.Label(
            logo_frame,
            text="Sovereign EDU",
            font=("Inter", 9),
            bg=self.colors['bg_sidebar'],
            fg=self.colors['text_secondary']
        )
        subtitle.pack()
        
        # Constitution section
        self.add_section_title(sidebar, "Constitución Activa")
        self.add_axiom_item(sidebar, "✓ Aritmética de Peano", active=True)
        self.add_axiom_item(sidebar, "✓ Álgebra Lineal", active=True)
        self.add_axiom_item(sidebar, "✓ Geometría Euclidiana", active=True)
        
        self.add_section_title(sidebar, "Bloqueado")
        self.add_axiom_item(sidebar, "✗ Cálculo Diferencial", active=False)
        self.add_axiom_item(sidebar, "✗ Física Cuántica", active=False)
        
        # Status at bottom
        status_frame = tk.Frame(sidebar, bg=self.colors['bg_sidebar'])
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)
        
        status_label = tk.Label(
            status_frame,
            text="● Sistema Activo",
            font=("Inter", 10),
            bg=self.colors['bg_sidebar'],
            fg=self.colors['accent']
        )
        status_label.pack()
        
    def add_section_title(self, parent, text):
        label = tk.Label(
            parent,
            text=text.upper(),
            font=("Inter", 9, "bold"),
            bg=self.colors['bg_sidebar'],
            fg=self.colors['text_secondary'],
            anchor='w'
        )
        label.pack(fill=tk.X, padx=15, pady=(15, 8))
        
    def add_axiom_item(self, parent, text, active=True):
        frame = tk.Frame(parent, bg=self.colors['bg_sidebar'])
        frame.pack(fill=tk.X, padx=12, pady=2)
        
        color = self.colors['text_primary'] if active else self.colors['text_secondary']
        
        label = tk.Label(
            frame,
            text=text,
            font=("Inter", 11),
            bg=self.colors['bg_sidebar'],
            fg=color,
            anchor='w',
            padx=10,
            pady=8
        )
        label.pack(fill=tk.X)
        
    def create_chat_area(self, parent):
        chat_container = tk.Frame(parent, bg=self.colors['bg_main'])
        chat_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Header
        header = tk.Frame(chat_container, bg=self.colors['bg_main'], height=60)
        header.pack(fill=tk.X, padx=20, pady=(10, 0))
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text="Tutor de Matemáticas (Grado 10)",
            font=("Inter", 14, "bold"),
            bg=self.colors['bg_main'],
            fg=self.colors['text_primary']
        )
        title.pack(side=tk.LEFT, pady=15)
        
        # Messages area (scrollable)
        self.messages_frame = ModernScrollableFrame(chat_container)
        self.messages_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Input area
        self.create_input_area(chat_container)
        
    def create_input_area(self, parent):
        input_container = tk.Frame(parent, bg=self.colors['bg_main'])
        input_container.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Input frame with border
        input_frame = tk.Frame(
            input_container, 
            bg=self.colors['bg_main'],
            highlightbackground=self.colors['border'],
            highlightthickness=1,
            highlightcolor=self.colors['accent']
        )
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Text input
        self.input_text = tk.Text(
            input_frame,
            height=3,
            font=("Inter", 12),
            bg=self.colors['bg_main'],
            fg=self.colors['text_primary'],
            relief=tk.FLAT,
            wrap=tk.WORD,
            padx=15,
            pady=12,
            insertbackground=self.colors['text_primary']  # Cursor color
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind events
        self.input_text.bind('<Return>', self.on_enter_key)
        self.input_text.bind('<Shift-Return>', lambda e: None)  # Allow Shift+Enter for newline
        
        # Focus on input by default
        self.input_text.focus_set()
        
        # Send button
        send_btn = tk.Button(
            input_frame,
            text="↑",
            font=("Inter", 16, "bold"),
            bg=self.colors['accent'],
            fg='white',
            relief=tk.FLAT,
            width=3,
            cursor='hand2',
            command=self.send_message,
            activebackground='#0d8c6a',
            activeforeground='white'
        )
        send_btn.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Disclaimer
        disclaimer = tk.Label(
            input_container,
            text="CMFO no alucina. Cada respuesta está verificada estructuralmente.",
            font=("Inter", 9),
            bg=self.colors['bg_main'],
            fg=self.colors['text_secondary']
        )
        disclaimer.pack()
        
    def on_enter_key(self, event):
        # Check if Shift is pressed
        if event.state & 0x1:  # Shift key
            return  # Allow newline
        else:
            self.send_message()
            return 'break'  # Prevent default newline
        
    def show_welcome(self):
        welcome_frame = tk.Frame(
            self.messages_frame.scrollable_frame,
            bg=self.colors['bg_main']
        )
        welcome_frame.pack(pady=100, padx=50)
        
        title = tk.Label(
            welcome_frame,
            text="¿En qué puedo ayudarte?",
            font=("Inter", 24, "bold"),
            bg=self.colors['bg_main'],
            fg=self.colors['text_primary']
        )
        title.pack(pady=(0, 30))
        
        # Example buttons
        examples = [
            ("📐", "Ecuaciones lineales", "Explícame ecuaciones lineales"),
            ("🔍", "Verificar axioma", "¿Es correcto que (a+b)² = a² + b²?"),
            ("🚫", "Tema bloqueado", "Explica derivadas")
        ]
        
        for icon, title_text, query in examples:
            self.create_example_button(welcome_frame, icon, title_text, query)
            
    def create_example_button(self, parent, icon, title, query):
        btn_frame = tk.Frame(
            parent,
            bg='white',
            highlightbackground=self.colors['border'],
            highlightthickness=1,
            cursor='hand2'
        )
        btn_frame.pack(fill=tk.X, pady=5)
        btn_frame.bind('<Button-1>', lambda e: self.quick_send(query))
        
        content = tk.Frame(btn_frame, bg='white')
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        icon_label = tk.Label(
            content,
            text=icon,
            font=("Inter", 18),
            bg='white'
        )
        icon_label.pack(side=tk.LEFT, padx=(0, 10))
        icon_label.bind('<Button-1>', lambda e: self.quick_send(query))
        
        text_frame = tk.Frame(content, bg='white')
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_frame.bind('<Button-1>', lambda e: self.quick_send(query))
        
        title_label = tk.Label(
            text_frame,
            text=title,
            font=("Inter", 11, "bold"),
            bg='white',
            fg=self.colors['text_primary'],
            anchor='w'
        )
        title_label.pack(fill=tk.X)
        title_label.bind('<Button-1>', lambda e: self.quick_send(query))
        
    def quick_send(self, text):
        self.input_text.delete('1.0', tk.END)
        self.input_text.insert('1.0', text)
        self.send_message()
        
    def send_message(self):
        # Get message and strip whitespace
        message = self.input_text.get('1.0', 'end-1c').strip()
        
        if not message:
            return
            
        # Clear input immediately
        self.input_text.delete('1.0', tk.END)
        self.input_text.focus_set()
        
        # Clear welcome if present
        for widget in self.messages_frame.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Add user message
        self.add_message(message, is_user=True)
        
        # Show typing indicator
        typing_frame = self.add_typing_indicator()
        
        # Process in background thread
        def process():
            try:
                print(f"[DEBUG] Processing query: {message}")
                result = self.tutor.interact(message)
                response = result["response"]["response"]
                status = result["response"]["status"]
                
                print(f"[DEBUG] Got response: {response[:50]}... Status: {status}")
                
                # Put result in queue
                self.response_queue.put({
                    'type': 'response',
                    'typing_frame': typing_frame,
                    'response': response,
                    'status': status
                })
                
            except Exception as e:
                print(f"[ERROR] {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Put error in queue
                self.response_queue.put({
                    'type': 'error',
                    'typing_frame': typing_frame,
                    'message': str(e)
                })
                
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
        
        # Start checking queue
        self.check_response_queue()
        
    def check_response_queue(self):
        """Check queue for responses from background thread"""
        try:
            while True:
                msg = self.response_queue.get_nowait()
                
                # Remove typing indicator
                if 'typing_frame' in msg and msg['typing_frame']:
                    msg['typing_frame'].destroy()
                
                # Handle response
                if msg['type'] == 'response':
                    self.add_message(msg['response'], is_user=False, status=msg['status'])
                    print("[DEBUG] Message added to UI")
                elif msg['type'] == 'error':
                    self.add_message(f"Error: {msg['message']}", is_user=False, status="ERROR")
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_response_queue)
        
    def add_typing_indicator(self):
        frame = tk.Frame(
            self.messages_frame.scrollable_frame,
            bg=self.colors['bg_main']
        )
        frame.pack(fill=tk.X, pady=10, padx=20)
        
        # Avatar
        avatar = tk.Label(
            frame,
            text="C",
            font=("Inter", 12, "bold"),
            bg=self.colors['accent'],
            fg='white',
            width=3,
            height=1
        )
        avatar.pack(side=tk.LEFT, padx=(0, 10))
        
        # Typing animation
        typing_label = tk.Label(
            frame,
            text="●●●",
            font=("Inter", 14),
            bg=self.colors['bg_main'],
            fg=self.colors['text_secondary']
        )
        typing_label.pack(side=tk.LEFT)
        
        self.messages_frame.scroll_to_bottom()
        return frame
        
    def remove_typing_indicator(self, frame):
        frame.destroy()
        
    def add_message(self, text, is_user=False, status=None):
        msg_container = tk.Frame(
            self.messages_frame.scrollable_frame,
            bg=self.colors['bg_user'] if is_user else self.colors['bg_main']
        )
        msg_container.pack(fill=tk.X, pady=5)
        
        # Inner frame for padding
        inner = tk.Frame(
            msg_container,
            bg=self.colors['bg_user'] if is_user else self.colors['bg_main']
        )
        inner.pack(fill=tk.X, padx=20, pady=15)
        
        # Avatar
        avatar_bg = '#5436da' if is_user else self.colors['accent']
        avatar_text = "Tú" if is_user else "C"
        
        avatar = tk.Label(
            inner,
            text=avatar_text,
            font=("Inter", 11, "bold"),
            bg=avatar_bg,
            fg='white',
            width=4,
            height=2
        )
        avatar.pack(side=tk.LEFT, padx=(0, 15), anchor='n')
        
        # Message content
        content_frame = tk.Frame(
            inner,
            bg=self.colors['bg_user'] if is_user else self.colors['bg_main']
        )
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        message_label = tk.Label(
            content_frame,
            text=text,
            font=("Inter", 12),
            bg=self.colors['bg_user'] if is_user else self.colors['bg_main'],
            fg=self.colors['text_primary'],
            wraplength=700,
            justify=tk.LEFT,
            anchor='w'
        )
        message_label.pack(fill=tk.X, pady=(5, 0))
        
        # Status badge for assistant
        if not is_user and status:
            badge_text = "✓ Verificado"
            badge_color = self.colors['accent']
            
            if status in ['BLOCKED', 'ERROR']:
                badge_text = "⚠ Bloqueado"
                badge_color = '#ef4444'
            elif status == 'AXIOM_VIOLATION':
                badge_text = "✗ Violación Axiomática"
                badge_color = '#ef4444'
                
            badge = tk.Label(
                content_frame,
                text=badge_text,
                font=("Inter", 9, "bold"),
                bg=self.colors['bg_main'],
                fg=badge_color
            )
            badge.pack(anchor='w', pady=(8, 0))
            
        self.messages_frame.scroll_to_bottom()

def main():
    root = tk.Tk()
    app = CMFODesktopApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
