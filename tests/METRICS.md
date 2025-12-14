# Evaluation Metrics

These metrics are enforced by the CI pipeline to ensure the "Maximum Level" standard of the CMFO engine.

## 1. Determinism
**Definition:** The engine must produce bit-exact identical outputs for identical inputs across all runs and platforms.
- **Metric:** `Variance(Run_A, Run_B) == 0`
- **Tolerance:** 0 bits.

## 2. Reversibility (Information Conservation)
**Definition:** For any geometric operation $Op$, applying the inverse $Op^{-1}$ must yield the original state.
- **Metric:** $|| \Psi_{initial} - Op^{-1}(Op(\Psi_{initial})) || < \epsilon$
- **Tolerance:** $\epsilon < 10^{-15}$ (Float64 machine epsilon).

## 3. Physics Precision
**Definition:** Derived physical constants must match CODATA experimental values within 10% (pure geometric) or 1% (with shape corrections).
- **Metric:** `abs(Derived - Experimental) / Experimental`
- **Tolerance:**
    - Pure Resonances (Muon): < 10%
    - Coupled Resonances (Electron): < 1%

## 4. Performance
**Definition:** The geometric approach must outperform brute-force methods for specific complexity classes (e.g., Inverse Hashing).
- **Metric:** `Time(Geometric) / Time(BruteForce)`
- **Target:** $< 10^{-3}$ (1000x Speedup).
