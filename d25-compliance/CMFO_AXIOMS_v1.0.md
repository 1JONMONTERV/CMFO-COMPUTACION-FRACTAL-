# CMFO Register of Axiomatic Assumptions (Structural Bias) v1.0

**Document ID**: CMFO-AXIOMS-001
**Version**: 1.0.0
**Status**: ACTIVE
**Alignment**: ISO/IEC 24027 (Bias in AI systems)

## 1. Introduction
CMFO is a deterministic system. It does not possess statistical bias derived from training data. However, it possesses **structural bias** derived from its axioms. This document declares these biases explicitly to Ensure transparency and auditability.

## 2. Core Axiomatic Biases

### A1. The Axiom of Structural Truth
> **Bias**: CMFO inherently privileges structured, demonstrative knowledge over intuitive, revealed, or consensus-based knowledge.
> **Impact**: High-validity intuitive claims (e.g., indigenous knowledge without formal proofs) may be classified as `SPECULATIVE` or `UNVERIFIED`.
> **Mitigation**: The system must explicitly label such claims as `SPECULATIVE` rather than `FALSE`, acknowledging the limitation of the method, not the falsehood of the claim.

### A2. The Axiom of Domain Sovereignty
> **Bias**: CMFO assumes knowledge can be cleanly separated into distinct domains (Physics, Theology, Math).
> **Impact**: Hybrid or holistic claims that blend domains metaphorically may be rejected as `INVALID` (Cross-Domain Contamination).
> **Mitigation**: Users are educated via the Interface Layer to separate complex queries into atomic, domain-specific components.

### A3. The Fractal Threshold Axiom ($\phi$)
> **Bias**: CMFO uses the Golden Ratio ($\phi \approx 1.618$) as the threshold for Truth and $\phi^{-1} \approx 0.618$ for Falsity in fuzzy logic operations.
> **Impact**: This is a specific design choice involving non-linear logic. It differs from Boolean (0/1) or Probabilistic (0.0-1.0) logic standards.
> **Mitigation**: All audit traces export the raw values so external auditors can map them to other logic systems if needed.

## 3. Explicit Exclusions
CMFO is structurally incapable of determining truth in the following areas:
1.  **Aesthetics**: "Is this painting beautiful?" (Subjective).
2.  **Personal Intent**: "Did X mean to hurt Y?" (Unobservable mental state).
3.  **Future Prediction (Open Systems)**: "Who will win the election?" (Complex chaotic system without closed-form solution).

## 4. Axioms Provenance (Audit Trail)
Each axiom is documented with its historical motivation and failure modes.

### A1. The Axiom of Structural Truth
*   **Motivation**: To prevent "Argument from Authority". Truth must be intrinsic to the object, not the subject.
*   **Scope**: All 12 Domains.
*   **Limitations**: Cannot validate hermetic or esoteric knowledge where structure is deliberately hidden.
*   **Failure Mode**: May reject valid heuristic solutions before they are formalized.

### A2. The Axiom of Domain Sovereignty
*   **Motivation**: To prevent category errors (e.g., measuring God with a ruler, or measuring electrons with scripture).
*   **Scope**: Inter-domain translation.
*   **Limitations**: Borderline cases (e.g., Bio-Ethics) require careful manually bridged axioms.
*   **Failure Mode**: "Siloing" of knowledge if bridges are not defined (D22 fixes this via Tension Metrics).

### A3. The Fractal Threshold Axiom ($\phi$)
*   **Motivation**: Binary logic (0/1) is too rigid; Probability (0-100) is too fuzzy. Fractal logic offers stability.
*   **Scope**: Conflict resolution.
*   **Limitations**: Computational overhead of high-precision floating point comparisons.
*   **Failure Mode**: "Edge of Chaos" detections in highly recursive loops.

## 5. Certification of Determinism
The undersigned architect certifies that for any strictly identical input pair (Context, Query), the system produces identical outputs.

**Signed**: *CMFO Architecture Team*
