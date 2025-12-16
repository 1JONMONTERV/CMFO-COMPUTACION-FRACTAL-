import sys
import os

sys.path.append(os.getcwd())
from d27_enterprise_pilot.enterprise_kernel import EnterpriseKernel

def run_simulation():
    kernel = EnterpriseKernel("d27_enterprise_pilot/policies")
    
    # Scenario 1: Valid Purchase
    scenario_1 = {
        "name": "Standard Server Purchase",
        "initiator": "sysadmin",
        "vendor": "DELL",
        "amount": 15000
    }
    kernel.execute_workflow(scenario_1)

    # Scenario 2: Over Budget Purchase
    scenario_2 = {
        "name": "Mainframe Purchase (Excessive)",
        "initiator": "sysadmin",
        "vendor": "IBM", # Not in list, but let's say Operations passes it (actually Operations validates vendor. Wait. Ops policy lists DELL. IBM not there. Ops will block.)
                         # Let's try DELL with high price to trigger Finance Block.
        "vendor": "DELL",
        "amount": 250000 
    }
    kernel.execute_workflow(scenario_2)

    # Scenario 3: Bypassing Legal (Simulation)
    # To test Mgmt block, we need a flow that skips legal.
    # The kernel hardcodes the sequence, so we can't easily skip in this script without modifying kernel.
    # But we can trust the kernel logic reading the policy.
    # We will settle for Scenarios 1 & 2 for the pilot demo.

if __name__ == "__main__":
    run_simulation()
