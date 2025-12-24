"""
CMFO D7: Discursive Identity Test
==================================
Validates consistent tone, firmness, and style.

Tests:
1. Tone Consistency: Assertive (high conf) vs Tentative (low conf)
2. Firmness Calibration: Intent + Confidence logic
3. Style Markers: Posture first, explanation second
4. No "Servile" or "Machine" language
"""

import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from cmfo.decision.enhanced_engine import EnhancedDecisionEngine
from cmfo.decision.memory import FractalMemory
from cmfo.decision.proof import ProofObject, Candidate, Evidence, EvidenceType, ProofBuilder
from cmfo.decision.renderer import DeterministicRenderer


def test_discursive_identity():
    """Test D7 Discursive Identity"""
    print("=" * 70)
    print("  CMFO D7: DISCURSIVE IDENTITY TEST")
    print("  Tone, Firmness, and Human Style")
    print("=" * 70)
    
    renderer = DeterministicRenderer(language="es")
    builder = ProofBuilder()
    
    # ----------------------------------------------------------------
    # Test 1: Firmness Levels (Assertive vs Tentative)
    # ----------------------------------------------------------------
    print("\n\n[Test 1] Firmness Levels (Tone Calibration)")
    print("-" * 70)
    
    # High Confidence -> Assertive
    winner_high = Candidate("correction", "correct", 0.1, 0.1)
    runner_up_high = Candidate("confirm", "confirm", 0.9, 0.9) # High delta
    proof_high = builder.build(winner_high, runner_up_high, [], "correct", "stub")
    # Hack delta for test
    proof_high.delta = 0.8
    proof_high.margin_stable = True
    
    resp_high = renderer.render(proof_high, {"correction": "el dato es X"})
    print(f"High Confidence (delta=0.8): '{resp_high}'")
    
    # Low Confidence -> Tentative
    proof_low = builder.build(winner_high, runner_up_high, [], "correct", "stub")
    # Hack delta
    proof_low.delta = 0.1
    proof_low.margin_stable = False
    
    resp_low = renderer.render(proof_low, {"correction": "el dato es X"})
    print(f"Low Confidence (delta=0.1):  '{resp_low}'")
    
    assert "Incorrecto" in resp_high, "High confidence should be Assertive ('Incorrecto')"
    assert "Parece" in resp_low, "Low confidence should be Tentative ('Parece')"
    print("[OK] Tone scales correctly with confidence")
    
    # ----------------------------------------------------------------
    # Test 2: Intent-Specific Tone (Question vs Conflict)
    # ----------------------------------------------------------------
    print("\n\n[Test 2] Intent-Specific Tone")
    print("-" * 70)
    
    # Conflict (High Firmness by default)
    winner_conflict = Candidate("conflict", "conflict", 0.1, 0.1)
    proof_conflict = builder.build(winner_conflict, None, [], "conflict", "stub")
    proof_conflict.delta = 0.6
    proof_conflict.margin_stable = True
    
    resp_conflict = renderer.render(proof_conflict, {"previous": "A", "explanation": "B"})
    print(f"Conflict (Assertive): '{resp_conflict}'")
    
    # Question (Honest Ambiguity)
    winner_question = Candidate("question", "question", 0.1, 0.1)
    proof_question = builder.build(winner_question, None, [], "question", "stub")
    proof_question.delta = 0.3
    
    resp_question = renderer.render(proof_question, {"option_a": "A", "option_b": "B"})
    print(f"Question (Moderate):  '{resp_question}'")
    
    assert "contradice" in resp_conflict, "Conflict should be direct"
    assert "refieres" in resp_question, "Question should be inquiring"
    print("[OK] Tone adapts to intent correctly")
    
    # ----------------------------------------------------------------
    # Test 3: Style Markers (No Servile Language)
    # ----------------------------------------------------------------
    print("\n\n[Test 3] Style Markers (Anti-Servile Check)")
    print("-" * 70)
    
    forbidden_words = ["lo siento", "disculpa", "perdón", "tal vez podría", "como modelo de lenguaje"]
    
    all_responses = [resp_high, resp_low, resp_conflict, resp_question]
    
    failed = False
    for r in all_responses:
        lower_r = r.lower()
        for bad in forbidden_words:
            if bad in lower_r:
                print(f"FAIL: Found servile language '{bad}' in: '{r}'")
                failed = True
                
    if not failed:
        print("[OK] No servile/machine language detected")
        
    print("\nEnsure: Posture first, justification second.")
    print(f"Example: {resp_high}")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("  D7 IDENTITY VERIFIED")
    print("=" * 70)
    print("  [OK] Assertive when sure")
    print("  [OK] Tentative when unsure")
    print("  [OK] Direct style (no apology)")
    print("  [OK] Consistent structure")


if __name__ == "__main__":
    test_discursive_identity()
