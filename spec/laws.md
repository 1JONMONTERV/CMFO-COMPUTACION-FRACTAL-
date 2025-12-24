# CMFO Fundamental Laws

## L1: Closure Law

**Statement:**
```
∀ op ∈ Ω, ∀ x,y ∈ X: op(x,y) ∈ X
```

**Meaning:** All grammatical operations produce valid semantic vectors.

**Test:** `tests/test_invariants.py::test_closure`

---

## L2: Norm Invariance Law

**Statement:**
```
∀ op ∈ Ω, ∀ x,y ∈ X: ||op(x,y)||_φ = 1 ± ε_norm
```

Where ε_norm = 0.01

**Meaning:** Operations preserve semantic magnitude (unit norm).

**Test:** `tests/test_invariants.py::test_norm_preservation`

---

## L3: Non-Commutativity Law (Word Order)

**Statement:**
```
APP_s(APP_o(v,o),s) ≠_ε APP_s(APP_o(v,s),o)
```

Where ≠_ε means d_φ(...) > ε_diff (ε_diff = 0.05)

**Meaning:** "Juan ve a María" ≠ "María ve a Juan"

**Test:** `tests/test_laws_composition.py::test_word_order`

---

## L4: Approximate Idempotence (Modification)

**Statement:**
```
d_φ(MOD(MOD(n,a),a), MOD(n,a)) < ε_idem
```

Where ε_idem = 0.1

**Meaning:** "casa roja roja" ≈ "casa roja" (redundant modification)

**Test:** `tests/test_laws_composition.py::test_idempotence`

---

## L5: Typed Associativity (Approximate)

**Statement:**
```
For compatible types:
d_φ(op1(op2(x,y),z), op2(x,op1(y,z))) < ε_assoc
```

Where ε_assoc = 0.15

**Meaning:** Grouping doesn't drastically change meaning (when types allow).

**Test:** `tests/test_laws_composition.py::test_associativity`

---

## L6: Concordance Symmetry

**Statement:**
```
CONC_gen(n,a) = CONC_gen(a,n)  (up to ε)
```

**Meaning:** "perro negro" ≈ "negro perro" (both valid, same semantics)

**Test:** `tests/test_laws_composition.py::test_concordance`

---

## L7: Temporal Composition

**Statement:**
```
T_perf(T_past(v)) ≈_ε T_pqp(v)
```

Where T_pqp = pluscuamperfecto (past perfect)

**Meaning:** Temporal operators compose predictably.

**Test:** `tests/test_laws_composition.py::test_temporal_composition`

---

## L8: Negation Involution

**Statement:**
```
NEG(NEG(x)) ≈_ε x
```

**Meaning:** Double negation returns to original (approximately).

**Test:** `tests/test_invariants.py::test_negation_involution`

---

## L9: Locality of Span

**Statement:**
```
∀ op ∈ Ω, ∀ x,y ∈ X:
span(op(x,y)) ⊆ span({x,y}) + ε_span
```

**Meaning:** Result stays "near" the subspace of inputs.

**Test:** `tests/test_invariants.py::test_span_locality`

---

## L10: Convergence Under Iteration

**Statement:**
```
For stable operators op:
∃ A_i: lim_{n→∞} op^n(x) ∈ Basin(A_i)
```

**Meaning:** Repeated application converges to attractor (not diverges).

**Test:** `tests/test_attractors_metric.py::test_convergence_stability`

---

## Verification Protocol

**Every law must have:**
1. Mathematical statement (above)
2. Automated test (in `tests/`)
3. Tolerance parameters (ε values)
4. Failure examples (edge cases)

**CI runs all tests on:**
- Every commit
- Every PR
- Nightly (full suite)

**Failure = Breaking change**
- Law violation blocks merge
- No "it works in theory" exceptions
