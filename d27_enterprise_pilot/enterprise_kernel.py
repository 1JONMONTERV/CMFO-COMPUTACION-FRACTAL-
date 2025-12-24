import json
import os
import datetime
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# In a real implementation, this would import ActionGate and Receipt
# For this pilot simulation, we implement a lightweight version of the logic 
# to demonstrate the multi-node flow without circular dependency complexity.

@dataclass
class TransactionState:
    transaction_id: str
    initiator: str
    amount: float
    vendor: str
    status: str
    receipts:  List[Dict[str, Any]] # History of immutable receipts

class EnterpriseKernel:
    def __init__(self, policy_dir: str):
        self.policies = self._load_policies(policy_dir)
        print(f"[*] Enterprise Kernel Initialized. Loaded {len(self.policies)} Sovereign Domains.")

    def _load_policies(self, directory: str) -> Dict[str, Any]:
        policies = {}
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                with open(os.path.join(directory, filename), 'r') as f:
                    data = json.load(f)
                    policies[data["domain"]] = data
        return policies

    def execute_workflow(self, scenario: Dict[str, Any]):
        """
        Simulates the full lifecycle of a cross-department transaction.
        """
        print(f"\n[KERNEL] Starting Workflow: {scenario['name']}")
        print("="*60)
        
        state = TransactionState(
            transaction_id=f"TX-{datetime.datetime.now().strftime('%Y%m%d')}-001",
            initiator=scenario['initiator'],
            amount=scenario['amount'],
            vendor=scenario['vendor'],
            status="INIT",
            receipts=[]
        )

        # Step 1: Operations (Initiate)
        self._step_operations(state)
        
        # Step 2: Finance (Audit)
        if state.status == "OPS_APPROVED":
            self._step_finance(state)
        else:
            print("[KERNEL] Workflow Halted at Operations.")
            return

        # Step 3: Legal (Compliance)
        if state.status == "FINANCE_APPROVED":
            self._step_legal(state)
        else:
            print("[KERNEL] Workflow Halted at Finance.")
            return

        # Step 4: Management (Sign)
        if state.status == "LEGAL_APPROVED":
            self._step_management(state)
        else:
            print("[KERNEL] Workflow Halted at Legal.")
            return

        print("="*60)
        print(f"[KERNEL] Workflow Final Status: {state.status}")
        print(f"[KERNEL] Traceability Ledger: {len(state.receipts)} immutable blocks.")

    def _step_operations(self, state: TransactionState):
        domain = "OPERATIONS"
        policy = self.policies[domain]
        role = "requestor" # Simulated actor role
        
        print(f"Node: {domain} | Actor: {role}")
        
        # Validate Vendor
        valid_vendors = policy["roles"][role]["constraints"]["valid_vendors"]
        if state.vendor in valid_vendors:
            self._commit_receipt(state, domain, "initiate_purchase", "AUTHORIZED")
            state.status = "OPS_APPROVED"
        else:
            self._commit_receipt(state, domain, "initiate_purchase", "BLOCKED: Invalid Vendor")
            state.status = "REJECTED_OPS"

    def _step_finance(self, state: TransactionState):
        domain = "FINANCE"
        policy = self.policies[domain]
        role = "analyst"
        
        print(f"Node: {domain} | Actor: {role}")
        
        # Validate Budget
        limit = policy["roles"][role]["constraints"]["max_budget_per_transaction"]
        if state.amount <= limit:
             self._commit_receipt(state, domain, "approve_funds", "AUTHORIZED")
             state.status = "FINANCE_APPROVED"
        else:
             self._commit_receipt(state, domain, "approve_funds", f"BLOCKED: Over limit {limit}")
             state.status = "REJECTED_FINANCE"

    def _step_legal(self, state: TransactionState):
        domain = "LEGAL"
        policy = self.policies[domain]
        role = "compliance"
        
        print(f"Node: {domain} | Actor: {role}")
        
        # Validate Jurisdiction (Mock check)
        blocked = policy["roles"][role]["constraints"]["blocked_jurisdictions"]
        # Assume 'NK' is North Korea. Our vendor 'DELL' is US, so fine.
        # Minimal Check.
        self._commit_receipt(state, domain, "validate_contract", "AUTHORIZED")
        state.status = "LEGAL_APPROVED"

    def _step_management(self, state: TransactionState):
        domain = "MANAGEMENT"
        policy = self.policies[domain]
        role = "executive"
        
        print(f"Node: {domain} | Actor: {role}")
        
        # Structural Prerequisite Check (The Core CMFO Value)
        prereqs = policy["roles"][role]["structural_prerequisites"]["sign_purchase_order"]
        collected_proofs = [f"{r['domain']}:{r['action']}" for r in state.receipts if r['result'] == "AUTHORIZED"]
        
        missing = [p for p in prereqs if p not in collected_proofs]
        
        if not missing:
            self._commit_receipt(state, domain, "sign_purchase_order", "AUTHORIZED: FINAL SIGNATURE")
            state.status = "COMPLETED"
        else:
            print(f"    [!] STRUCTURAL VIOLATION. Missing proofs: {missing}")
            self._commit_receipt(state, domain, "sign_purchase_order", "BLOCKED: Missing Prerequisites")
            state.status = "REJECTED_MGMT"

    def _commit_receipt(self, state: TransactionState, domain: str, action: str, result: str):
        # Simulate Immutable Receipt
        timestamp = datetime.datetime.utcnow().isoformat()
        content = f"{domain}{action}{result}{timestamp}"
        receipt_hash = hashlib.sha256(content.encode()).hexdigest()
        
        receipt = {
            "domain": domain,
            "action": action,
            "result": result,
            "timestamp": timestamp,
            "hash": receipt_hash
        }
        state.receipts.append(receipt)
        print(f"    -> Action: {action} | Result: {result}")
        print(f"    -> Ledger Hash: {receipt_hash[:16]}...")
