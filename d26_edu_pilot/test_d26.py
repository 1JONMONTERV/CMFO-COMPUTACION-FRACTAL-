from pathlib import Path
from curriculum_compiler import CurriculumCompiler
from validator import EpistemologicalValidator
import sys

# Force UTF-8 for console output
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

def run_test():
    print("[*] CMFO-EDU: Initializing Curriculum Compiler...")
    
    # 1. Compile
    compiler = CurriculumCompiler(Path("d26-edu-pilot/syllabus_math_10.json"))
    profile = compiler.compile()
    
    print(f"[+] Curriculum Compiled: {profile.domain_name}")
    print(f"    - Allowed Concepts: {len(profile.allowed_topics)}")
    print(f"    - Forbidden Concepts: {len(profile.forbidden_topics)}")
    print("-" * 50)

    # 2. Configure Validator
    validator = EpistemologicalValidator()
    validator.configure_curriculum(profile)

    # 3. Canonical Test Case
    student_query = "Calcula la derivada de f(x)=2x+1"
    print(f"Student Input: \"{student_query}\"")
    
    # We simulate semantic extraction identifying "derivada" as the core concept
    extracted_concept = "derivada" 
    
    result = validator.validate_concept(extracted_concept)
    
    import json
    print("\nCMFO Decision:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 4. Verify Success Criteria
    if result["status"] == "REJECTED" and result["reason"] == "OUT_OF_SCOPE":
        print("\n[SUCCESS] System correctly rejected out-of-scope concept with pedagogical redirect.")
    else:
        print("\n[FAILURE] System did not handle the concept as expected.")

if __name__ == "__main__":
    run_test()
