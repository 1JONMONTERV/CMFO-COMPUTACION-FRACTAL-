import sys
import os
import json

sys.path.append(os.getcwd())
try:
    from cmfo.compliance.reporter import ComplianceReporter
except ImportError:
    pass

def demo_compliance_report():
    print("[*] Generating CMFO Sovereign Compliance Report")
    print("=" * 60)
    
    # 1. Simulate Evidence (Audit Logs from D27/D26)
    # These mimic the JSON receipts stored by SecureTutor
    mock_logs = [
        {
            "timestamp": "2025-12-16T10:00:00Z",
            "tutor_decision": "AUTHORIZED", 
            "student_query": "Explain Linear Eq",
            "locked": True,
            "ciphertext": "a1b2c3d4..." 
        },
        {
            "timestamp": "2025-12-16T10:05:00Z",
            "tutor_decision": "AXIOM_VIOLATION", 
            "student_query": "1 = 2",
            "locked": True,
            "ciphertext": "e5f6g7h8..."
        },
        {
            "timestamp": "2025-12-16T10:10:00Z",
            "tutor_decision": "BLOCKED", 
            "student_query": "Explain Derivatives",
            "locked": True,
            "ciphertext": "i9j0k1l2..."
        }
    ]
    
    # 2. Instantiate Reporter
    reporter = ComplianceReporter(
        system_name="CMFO-EDU-CORE (v1.0)",
        constitution_name="Syllabus_Math_10_v2.json"
    )
    
    # 3. Generate
    report_md = reporter.generate_report(mock_logs)
    
    # 4. Output
    print(report_md)
    
    # Save to file for user inspection
    with open("compliance_report_demo.md", "w") as f:
        f.write(report_md)
        
    print("\n[SUCCESS] Report generated: compliance_report_demo.md")

if __name__ == "__main__":
    demo_compliance_report()
