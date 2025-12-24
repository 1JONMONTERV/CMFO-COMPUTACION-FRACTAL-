"""
CMFO Decision Validation
=========================
Test 5 cases: confirm, correct, question, reference, conflict.

All in semantic states, no text yet.
"""

import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from cmfo.decision.engine import DecisionEngine, Memory, SemanticState, d_phi


def test_case_1_confirm():
    """Case 1: Correct affirmation → confirmatory response"""
    engine = DecisionEngine()
    memory = Memory(states=[], attractors=[])
    
    # Input: positive, clear statement
    S_input = [0.8, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]
    
    response, metrics = engine.decide_response(S_input, memory)
    
    print("\n[Case 1] Correct Affirmation")
    print(f"  Input type: {engine.classify_input(S_input, memory)}")
    print(f"  Response: {response.label} ({response.type})")
    print(f"  d_input: {metrics['d_input']:.4f}")
    print(f"  Score: {metrics['total_score']:.4f}")
    
    return response, metrics


def test_case_2_correct():
    """Case 2: False affirmation → corrective response"""
    engine = DecisionEngine()
    memory = Memory(states=[], attractors=[])
    
    # Input: negative polarity (false)
    S_input = [-0.7, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0]
    
    response, metrics = engine.decide_response(S_input, memory)
    
    print("\n[Case 2] False Affirmation")
    print(f"  Input type: {engine.classify_input(S_input, memory)}")
    print(f"  Response: {response.label} ({response.type})")
    print(f"  d_input: {metrics['d_input']:.4f}")
    print(f"  Score: {metrics['total_score']:.4f}")
    
    return response, metrics


def test_case_3_question():
    """Case 3: Ambiguity → question"""
    engine = DecisionEngine()
    memory = Memory(states=[], attractors=[])
    
    # Input: low norm (ambiguous)
    S_input = [0.2, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
    
    response, metrics = engine.decide_response(S_input, memory)
    
    print("\n[Case 3] Ambiguous Input")
    print(f"  Input type: {engine.classify_input(S_input, memory)}")
    print(f"  Response: {response.label} ({response.type})")
    print(f"  d_input: {metrics['d_input']:.4f}")
    print(f"  Score: {metrics['total_score']:.4f}")
    
    return response, metrics


def test_case_4_reference():
    """Case 4: Repetition → memory reference"""
    engine = DecisionEngine()
    
    # Add previous state to memory
    prev_state = SemanticState(
        vector=[0.7, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
        label='previous_statement',
        type='affirmation'
    )
    memory = Memory(
        states=[prev_state],
        attractors=[[0.7, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]]
    )
    
    # Input: very similar to previous
    S_input = [0.72, 0.28, 0.12, 0.0, 0.0, 0.0, 0.0]
    
    response, metrics = engine.decide_response(S_input, memory)
    
    print("\n[Case 4] Repetition")
    print(f"  Input type: {engine.classify_input(S_input, memory)}")
    print(f"  Response: {response.label} ({response.type})")
    print(f"  d_input: {metrics['d_input']:.4f}")
    print(f"  d_memory: {metrics['d_memory']:.4f}")
    print(f"  Score: {metrics['total_score']:.4f}")
    
    return response, metrics


def test_case_5_conflict():
    """Case 5: Contradiction → conflict signal"""
    engine = DecisionEngine()
    
    # Memory with established attractor
    memory = Memory(
        states=[],
        attractors=[[0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    
    # Input: far from all attractors
    S_input = [-0.8, -0.5, 0.9, 0.0, 0.0, 0.0, 0.0]
    
    response, metrics = engine.decide_response(S_input, memory)
    
    print("\n[Case 5] Contradiction")
    print(f"  Input type: {engine.classify_input(S_input, memory)}")
    print(f"  Response: {response.label} ({response.type})")
    print(f"  d_input: {metrics['d_input']:.4f}")
    print(f"  Score: {metrics['total_score']:.4f}")
    
    return response, metrics


def main():
    print("=" * 60)
    print("  CMFO DECISION VALIDATION")
    print("  5 Test Cases (States Only)")
    print("=" * 60)
    
    results = []
    
    results.append(test_case_1_confirm())
    results.append(test_case_2_correct())
    results.append(test_case_3_question())
    results.append(test_case_4_reference())
    results.append(test_case_5_conflict())
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    for i, (response, metrics) in enumerate(results, 1):
        print(f"  Case {i}: {response.type:12s} (score={metrics['total_score']:.3f})")
    
    print("\n  All cases executed successfully.")
    print("  Decision engine is deterministic and geometric.")
    print("=" * 60)


if __name__ == "__main__":
    main()
