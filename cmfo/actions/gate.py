from typing import List, Dict, Optional
import datetime
import uuid
from .receipt import ActionReceipt

class ActionGate:
    """
    The Gatekeeper.
    Enforces the Authorization Matrix:
    - User Permission?
    - Domain Permission?
    - Proof Existence?
    - Legal/Ethical Block?
    """

    def __init__(self):
        # Simulation of legal restrictions (In prod, this would be a policy engine)
        self.legal_blocks = ["medical_diagnosis", "legal_advice"]

    def attempt_execution(
        self,
        action: str,
        domain: str,
        input_data: str,
        user_auth: bool,
        domain_auth: bool,
        proof_ref: Optional[str]
    ) -> ActionReceipt:
        
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        action_id = f"act-{uuid.uuid4()}"
        
        authorized_by = []
        blocked_by = []

        # 1. User Authorization
        if user_auth:
            authorized_by.append("USER")
        
        # 2. Domain Authorization
        if domain_auth:
            authorized_by.append(f"DOMAIN:{domain}")
        else:
            # If domain explicitly forbids it, logic depends on matrix.
            # For this prototype, lack of domain auth is neutral unless it's a "Domain Command"
            pass

        # 3. Proof Authorization
        if proof_ref:
            authorized_by.append(f"PROOF:{proof_ref}")

        # 4. Legal/Safety Blocks (The Veto)
        if action in self.legal_blocks:
            blocked_by.append("LEGAL:restricted_domain")

        # --- THE MATRIX LOGIC ---
        
        is_allowed = False
        
        if "LEGAL:restricted_domain" in blocked_by:
            is_allowed = False
            result_msg = "BLOCKED: Legal/Ethical Restriction."
        
        elif action == "execute_code":
            # Code execution: Needs User + Proof 
            # (Domain usually doesn't authorize arbitrary code unless strict)
            if user_auth and proof_ref:
                is_allowed = True
                result_msg = "EXECUTED: Code ran successfully."
            else:
                blocked_by.append("POLICY:code_needs_proof")
                result_msg = "BLOCKED: Code execution requires Proof and User auth."

        elif action == "solve_exercise":
             # Solving: Needs Domain OR Proof (and User)
             if user_auth and (domain_auth or proof_ref):
                 is_allowed = True
                 result_msg = "SOLVED: Exercise solution generated."
             else:
                 blocked_by.append("POLICY:unauthorized_solution")
                 result_msg = "BLOCKED: Solution not authorized by Domain or Proof."
        
        elif action == "derive_concept":
            # e.g. Derivative in 10th grade
            # If domain_auth is False (forbidden in syllabus), we block even if Proof exists.
            # Wait, the user said "If domain forbids, we block".
            # In our calling convention, 'domain_auth' represents "Is it allowed in domain?".
            # If 'domain_auth' is False for a specialized domain, it acts as a block.
            
            if domain_auth:
                 is_allowed = True
                 result_msg = "DERIVED: Concept derivation complete."
            else:
                 blocked_by.append(f"DOMAIN_RESTRICTION:{domain}")
                 result_msg = "BLOCKED: Concept forbidden in this domain."

        else:
            # Default safest policy
            if user_auth and proof_ref:
                is_allowed = True
                result_msg = "EXECUTED: General action."
            else:
                blocked_by.append("POLICY:default_safe")
                result_msg = "BLOCKED: Insufficient authorization."

        if not is_allowed:
            # Clear authorization if blocked, or keep them to show who *tried* to authorize?
            # User wants to know who authorized, but blocked_by overrides.
            pass

        return ActionReceipt(
            action_id=action_id,
            timestamp=timestamp,
            requested_by={"type": "human", "id": "current_user"},
            authorized_by=authorized_by,
            blocked_by=blocked_by,
            domain=domain,
            action=action,
            input_data=input_data,
            proof_ref=proof_ref,
            result=result_msg,
            confidence=1.0 if is_allowed else 0.0
        )
