# Algorithmic Impact Assessment (AIA) - CMFO v1.0

**Document ID**: CMFO-AIA-001
**Compliance Target**: EU AI Act (2023), UNESCO Rec. on Ethics of AI
**Risk Classification**: LOW (Transparency / Information System)

## 1. System Description
CMFO (Computación Matricial Fractal Ontológica) is a **knowledge arbitration and validation system**. It processes natural language queries to identify their semantic domain and validates them against internal structural rules.

**It is NOT**:
*   A social scoring system.
*   A biometric identification system.
*   A credit scoring engine.
*   A judicial decision-making tool.

## 2. Structural Bias (Non-Statistical)
**CMFO does not learn from data distributions or use statistical training arrays.**
It is NOT a machine learning system. Therefore, it is immune to "Dataset Bias" (e.g., race, gender skew in training data).

However, it possesses **Structural Bias**:
*   **Source**: Axiomatic choices and Domain Definitions.
*   **Nature**: It privileges logical/formal consistency over consensus or intuition.
*   **Impact**: It is intentionally "biased" towards scientific rigor and axiomatic coherence.
*   **Mitigation**: This bias is a feature, not a bug, designed to ensure epistemological safety. It is documented in `CMFO_AXIOMS_v1.0.md`.

## 3. Risk Assessment (EU AI Act)

### Prohibited Practices (Art. 5)
*   **Subliminal manipulation**: NO. CMFO is explicit and verifiable.
*   **Social scoring**: NO. CMFO evaluates claims, not people.
*   **Real-time remote biometrics**: NO.

### High-Risk Systems (Annex III)
*   **Biometrics**: N/A.
*   **Critical Infrastructure**: N/A (CMFO is software middleware).
*   **Education/Vocational Training**: N/A.
*   **Employment**: N/A.
*   **Essential Private Services (Credit)**: N/A.
*   **Law Enforcement**: N/A.

**Conclusion**: CMFO falls under **Limited Risk** or **Minimal Risk**, primarily requiring Transparency obligations (Art. 52), which are fully met by the 4-Layer Human Interface.

## 3. Impact on Fundamental Rights (UNESCO / Charter of Fundamental Rights)

### 3.1 Right to Non-Discrimination
*   **Risk**: Deterministic bias favoring formal education styles.
*   **Mitigation**: Interface labels distinguish "Formal Proof" from "Doctrinal Coherence", giving dignity to non-scientific domains (Theology, Philosophy) without confusing them with empirical science.

### 3.2 Right to Good Administration (Explicability)
*   **Feature**: Every output is traceable to a specific domain rule and proof check.
*   **Auditability**: The `AC_AUDIT_MODE` allows full introspection of the decision path.

### 3.3 Freedom of Expression
*   **Policy**: CMFO does not censor "false" claims; it labels them (e.g., `SPECULATIVE`, `UNVERIFIED`). This preserves the user's right to information while protecting the public interest in truth.

## 4. Monitoring Plan
*   **Quarterly**: Review of "Domain Error" logs to identify if legitimate new fields of knowledge are being rejected.
*   **Yearly**: Update of `CMFO_AXIOMS` to reflect any structural changes.

**Approval Date**: 2025-12-15
**Review Date**: 2026-12-15
