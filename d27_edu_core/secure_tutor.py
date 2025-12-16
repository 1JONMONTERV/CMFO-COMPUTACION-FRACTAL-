import sys
import os
import json
import datetime
from typing import Dict, List, Any

# Ensure imports work
sys.path.append(os.getcwd())

from d26_edu_pilot.tutor import PedagogicalTutor
from d26_edu_pilot.curricular_profile import CurricularProfile
from d26_edu_pilot.curriculum_compiler import CurriculumCompiler
from cmfo.security.audit_lock import AuditLock

class SecureTutor:
    """
    Phase 3A Core Engine.
    Wraps the Pedagogical Logic in a Sovereign Security Layer.
    """
    
    def __init__(self, syllabus_path: str, tutor_identity: str = "tutor_math_10"):
        self.identity = tutor_identity
        
        # 1. Compile Curriculum (Fractal Compression: JSON -> Immutable Profile)
        self.compiler = CurriculumCompiler(syllabus_path)
        self.profile: CurricularProfile = self.compiler.compile()
        
        # 2. Logic Engine
        self.logic_engine = PedagogicalTutor(syllabus_path) # Reusing D26 Logic
        
        # 3. Security Engine
        self.locker = AuditLock()
        
        self.session_log = []
        print(f"[*] SecureTutor Initialized. Identity: {self.identity}")
        print(f"[*] Governance Context: {self.profile.axioms}")

    def interact(self, student_query: str) -> Dict[str, Any]:
        """
        Process a query, generate response, and LOCK the audit trail.
        """
        # A. Execute Logic
        # The logic engine verifies the query against the syllabus
        # D26 Tutor method is 'process_student_query'
        engine_output = self.logic_engine.process_student_query(student_query)
        
        # Normalize Status for Logging
        status = "UNKNOWN"
        response_text = ""
        violation = None
        
        if engine_output["type"] == "response":
            status = "AUTHORIZED"
            response_text = engine_output["content"]
        elif engine_output["type"] == "block":
            status = "BLOCKED"
            response_text = engine_output["message"]
            violation = engine_output.get("authority")
        elif engine_output["type"] == "correction":
            status = "AXIOM_VIOLATION"
            response_text = engine_output["message"]
            violation = engine_output.get("authority")
        elif engine_output["type"] == "error":
            status = "ERROR"
            response_text = engine_output["message"]
        
        # B. Construct Log Entry
        # What actually happened?
        timestamp = datetime.datetime.utcnow().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "student_query": student_query,
            "tutor_decision": status,
            "tutor_response": response_text,
            "violation_detail": violation
        }
        
        # C. Structural Lock
        # We lock this log using the Tutor's ID and the AXIS of the syllabus (e.g. ['verdad', 'orden'])
        # Ideally we map syllabus axioms to 7D axioms. 
        # For this pilot, we use the raw axiom strings from the profile as context.
        context_axioms = self.profile.axioms 
        
        # NOTE: AuditLock expects axioms to be in the Semantic Algebra dictionary.
        # The syllabus has ["Peano Axioms", "Equality"].
        # Ideally we map these. For now, let's inject "verdad" and "orden" as the 
        # fundamental governance context to ensure lock works with current Algebra definitions.
        # In a full version, "Peano" would have a vector definition.
        
        governance_context = ["verdad", "orden", "entidad"] # The "Constitution" of the System
        
        locked_receipt = self.locker.lock_log(
            identity=self.identity,
            domain_context=governance_context,
            log_entry=log_entry
        )
        
        self.session_log.append(locked_receipt)
        
        return {
            "response": {"status": status, "response": response_text, "violation": violation},
            "receipt": locked_receipt
        }

    def verify_receipt(self, receipt_index: int) -> Dict[str, Any]:
        """
        Auditor Tool: Prove that a past interaction was legitimate.
        """
        if receipt_index >= len(self.session_log):
            return {"error": "Index out of bounds"}
            
        locked = self.session_log[receipt_index]
        
        # To unlock, acts as the Institution (using the same Identity + Constitution)
        governance_context = ["verdad", "orden", "entidad"]
        
        return self.locker.unlock_log(self.identity, governance_context, locked)
