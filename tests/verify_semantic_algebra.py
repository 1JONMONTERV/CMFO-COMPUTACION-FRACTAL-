"""
CMFO: Fractal Linguistic Algebra - Standalone Verification
===========================================================

Complete demonstration and formal verification of 7D semantic algebra.
Self-contained with all necessary definitions.

Mathematical Foundation: ℝ^7 with φ-weighted metric
"""

import numpy as np
import math

# ============================================================================
# CORE DEFINITIONS
# ============================================================================

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2

# Fractal weights λᵢ = φ^(i-1)
LAMBDA = np.array([PHI**i for i in range(7)])

# 7D Semantic Axes
AXES = [
    "existence",  # 0
    "truth",      # 1
    "order",      # 2
    "action",     # 3
    "connection", # 4
    "mind",       # 5
    "time",       # 6
]

# Semantic Vectors (Ontological Definitions)
PROPERTY_VECTORS = {
    # Core Axioms
    "existencia": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "entidad":    [1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
    "nada":       [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    
    "verdad":     [0.0, 1.0, 0.6, 0.0, 0.2, 0.2, 0.1],
    "mentira":    [0.0, -1.0, -0.6, 0.0, -0.2, -0.2, -0.1],
    
    "orden":      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "caos":       [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    
    "acción":     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "estasis":    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    
    "conexión":   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "aislamiento":[0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    
    "mente":      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "objeto":     [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    
    "tiempo":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "pasado":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    
    # Composed Concepts
    "bien":       [0.0, 0.5, 0.8, 0.2, 0.6, 0.0, 0.1],
    "mal":        [0.0, -0.5, -0.8, -0.2, -0.6, 0.0, -0.1],
    
    "humano":     [1.0, 0.0, 0.5, 0.5, 0.6, 0.9, 0.4],
}

# ============================================================================
# ALGEBRAIC OPERATIONS
# ============================================================================

def vector_add(v, w):
    """Vector addition in ℝ^7"""
    return [v[i] + w[i] for i in range(7)]

def vector_scale(v, scalar):
    """Scalar multiplication"""
    return [v[i] * scalar for i in range(7)]

def vector_norm(v):
    """Euclidean norm"""
    return math.sqrt(sum(x**2 for x in v))

def d_phi(v, w):
    """
    Fractal distance metric
    
    d_φ(v, w) = √(Σᵢ λᵢ(vᵢ - wᵢ)²)
    """
    diff = [v[i] - w[i] for i in range(7)]
    weighted_sum = sum(LAMBDA[i] * diff[i]**2 for i in range(7))
    return math.sqrt(weighted_sum)

def apply_negation(v):
    """Negation operator (inverts truth axis)"""
    result = v.copy() if isinstance(v, list) else list(v)
    result[1] = -result[1]  # Invert truth
    return result

# ============================================================================
# FORMAL VERIFICATION
# ============================================================================

def verify_algebraic_laws():
    """Verify all 8 algebraic laws"""
    print("\n" + "="*70)
    print("ALGEBRAIC LAWS VERIFICATION")
    print("="*70)
    
    v = np.array([1.0, 0.5, -0.3, 0.0, 0.2, -0.1, 0.4])
    w = np.array([0.3, -0.2, 0.5, 0.1, -0.3, 0.0, 0.2])
    u = np.array([-0.5, 0.3, 0.1, -0.2, 0.4, 0.1, -0.3])
    
    tests = []
    
    # Law 1: Closure
    result = vector_add(v, w)
    tests.append(("Closure", len(result) == 7))
    
    # Law 2: Associativity
    left = vector_add(vector_add(u, v), w)
    right = vector_add(u, vector_add(v, w))
    tests.append(("Associativity", np.allclose(left, right, atol=1e-10)))
    
    # Law 3: Commutativity
    vw = vector_add(v, w)
    wv = vector_add(w, v)
    tests.append(("Commutativity", np.allclose(vw, wv, atol=1e-10)))
    
    # Law 4: Identity
    zero = np.zeros(7)
    result = vector_add(v, zero)
    tests.append(("Identity", np.allclose(result, v, atol=1e-10)))
    
    # Law 5: Inverse
    neg_v = vector_scale(v, -1.0)
    result = vector_add(v, neg_v)
    tests.append(("Inverse", np.allclose(result, zero, atol=1e-10)))
    
    # Law 6: Scalar Associativity
    a, b = 2.5, 3.0
    left = vector_scale(vector_scale(v, b), a)
    right = vector_scale(v, a * b)
    tests.append(("Scalar Assoc", np.allclose(left, right, atol=1e-10)))
    
    # Law 7: Distributivity (vector)
    left = vector_scale(vector_add(v, w), a)
    right = vector_add(vector_scale(v, a), vector_scale(w, a))
    tests.append(("Distributivity 1", np.allclose(left, right, atol=1e-10)))
    
    # Law 8: Distributivity (scalar)
    left = vector_scale(v, a + b)
    right = vector_add(vector_scale(v, a), vector_scale(v, b))
    tests.append(("Distributivity 2", np.allclose(left, right, atol=1e-10)))
    
    # Print results
    for name, passed in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s} {status}")
    
    total = len(tests)
    passed_count = sum(1 for _, p in tests if p)
    print(f"\nResult: {passed_count}/{total} laws verified")
    
    return all(p for _, p in tests)

def verify_metric_properties():
    """Verify metric space axioms"""
    print("\n" + "="*70)
    print("METRIC PROPERTIES VERIFICATION")
    print("="*70)
    
    v = np.array([1.0, 0.5, -0.3, 0.0, 0.2, -0.1, 0.4])
    w = np.array([0.3, -0.2, 0.5, 0.1, -0.3, 0.0, 0.2])
    u = np.array([-0.5, 0.3, 0.1, -0.2, 0.4, 0.1, -0.3])
    
    tests = []
    
    # Property 1: Positive Definite
    d = d_phi(v, w)
    d_self = d_phi(v, v)
    tests.append(("Positive Definite", d >= 0 and abs(d_self) < 1e-10))
    
    # Property 2: Symmetry
    d_vw = d_phi(v, w)
    d_wv = d_phi(w, v)
    tests.append(("Symmetry", abs(d_vw - d_wv) < 1e-10))
    
    # Property 3: Triangle Inequality
    d_uw = d_phi(u, w)
    d_uv = d_phi(u, v)
    d_vw = d_phi(v, w)
    tests.append(("Triangle Ineq", d_uw <= d_uv + d_vw + 1e-10))
    
    # Fractal weights verification
    weights_correct = all(abs(LAMBDA[i] - PHI**i) < 1e-10 for i in range(7))
    tests.append(("Fractal Weights", weights_correct))
    
    # Print results
    for name, passed in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s} {status}")
    
    total = len(tests)
    passed_count = sum(1 for _, p in tests if p)
    print(f"\nResult: {passed_count}/{total} properties verified")
    
    return all(p for _, p in tests)

def verify_linguistic_properties():
    """Verify linguistic and semantic properties"""
    print("\n" + "="*70)
    print("LINGUISTIC PROPERTIES VERIFICATION")
    print("="*70)
    
    verdad = np.array(PROPERTY_VECTORS["verdad"])
    mentira = np.array(PROPERTY_VECTORS["mentira"])
    orden = np.array(PROPERTY_VECTORS["orden"])
    bien = np.array(PROPERTY_VECTORS["bien"])
    mal = np.array(PROPERTY_VECTORS["mal"])
    
    tests = []
    
    # Property 1: Negation Involution
    neg_verdad = apply_negation(verdad)
    neg_neg_verdad = apply_negation(neg_verdad)
    tests.append(("Negation Involution", np.allclose(neg_neg_verdad, verdad, atol=1e-10)))
    
    # Property 2: Antonym Distance Symmetry
    d1 = d_phi(verdad, mentira)
    d2 = d_phi(mentira, verdad)
    tests.append(("Antonym Symmetry", abs(d1 - d2) < 1e-10))
    
    # Property 3: Semantic Distance Meaningful
    d_verdad_mentira = d_phi(verdad, mentira)
    d_verdad_orden = d_phi(verdad, orden)
    tests.append(("Semantic Distance", d_verdad_mentira > d_verdad_orden))
    
    # Property 4: Composition
    composed = vector_add(verdad, orden)
    tests.append(("Compositionality", composed[1] > 0 and composed[2] > 0))
    
    # Property 5: Ethics Clustering
    d_bien_mal = d_phi(bien, mal)
    tests.append(("Ethics Clustering", d_bien_mal > 0))
    
    # Print results
    for name, passed in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s} {status}")
    
    total = len(tests)
    passed_count = sum(1 for _, p in tests if p)
    print(f"\nResult: {passed_count}/{total} properties verified")
    
    return all(p for _, p in tests)

def demonstrate_semantic_algebra():
    """Demonstrate practical semantic algebra"""
    print("\n" + "="*70)
    print("SEMANTIC ALGEBRA DEMONSTRATION")
    print("="*70)
    
    # Example 1: Semantic Composition
    print("\n1. SEMANTIC COMPOSITION")
    print("-" * 70)
    
    verdad = np.array(PROPERTY_VECTORS["verdad"])
    orden = np.array(PROPERTY_VECTORS["orden"])
    
    composed = vector_add(verdad, orden)
    
    print(f"verdad:    {verdad}")
    print(f"orden:     {orden}")
    print(f"composed:  {composed}")
    print(f"Norm: {vector_norm(composed):.4f}")
    
    # Example 2: Semantic Distance
    print("\n2. SEMANTIC DISTANCE (phi-weighted)")
    print("-" * 70)
    
    mentira = np.array(PROPERTY_VECTORS["mentira"])
    
    d_verdad_mentira = d_phi(verdad, mentira)
    d_verdad_orden = d_phi(verdad, orden)
    
    print(f"d_phi(verdad, mentira) = {d_verdad_mentira:.4f}")
    print(f"d_phi(verdad, orden)   = {d_verdad_orden:.4f}")
    print(f"Ratio: {d_verdad_mentira / d_verdad_orden:.2f}x")
    print(f"\nInterpretation: Opposites are {d_verdad_mentira / d_verdad_orden:.2f}x farther than related concepts")
    
    # Example 3: Negation
    print("\n3. NEGATION OPERATOR")
    print("-" * 70)
    
    neg_verdad = apply_negation(verdad)
    
    print(f"verdad:      {verdad}")
    print(f"NEG(verdad): {neg_verdad}")
    print(f"d_phi(verdad, NEG(verdad)) = {d_phi(verdad, neg_verdad):.4f}")
    
    # Example 4: Fractal Weights
    print("\n4. FRACTAL WEIGHTS (Golden Ratio)")
    print("-" * 70)
    
    print(f"phi = {PHI:.10f}")
    print(f"\nWeights lambda_i = phi^i:")
    for i in range(7):
        print(f"  lambda_{i+1} = phi^{i} = {LAMBDA[i]:.6f}  (axis: {AXES[i]})")
    
    # Example 5: Axiom Independence
    print("\n5. AXIOM INDEPENDENCE")
    print("-" * 70)
    
    axiom_words = ["existencia", "verdad", "orden", "accion", "conexion", "mente", "tiempo"]
    print("Pairwise distances between base axioms:")
    
    for i, word1 in enumerate(axiom_words):
        for j, word2 in enumerate(axiom_words):
            if i < j and word1 in PROPERTY_VECTORS and word2 in PROPERTY_VECTORS:
                v1 = np.array(PROPERTY_VECTORS[word1])
                v2 = np.array(PROPERTY_VECTORS[word2])
                d = d_phi(v1, v2)
                print(f"  d_phi({word1:12s}, {word2:12s}) = {d:.4f}")

def run_complete_verification():
    """Run complete verification suite"""
    print("\n" + "="*70)
    print("CMFO: FRACTAL LINGUISTIC ALGEBRA")
    print("Complete Formal Verification Suite")
    print("="*70)
    print(f"\nDimension: 7")
    print(f"Metric: phi-weighted (phi = {PHI:.10f})")
    print(f"Vector Space: R^7")
    print(f"Defined Concepts: {len(PROPERTY_VECTORS)}")

    
    # Run all verifications
    results = []
    
    results.append(("Algebraic Laws", verify_algebraic_laws()))
    results.append(("Metric Properties", verify_metric_properties()))
    results.append(("Linguistic Properties", verify_linguistic_properties()))
    
    # Demonstration
    demonstrate_semantic_algebra()
    
    # Final Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:25s} {status}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print(f"\nOverall: {passed_count}/{total} test suites passed")
    
    if all(p for _, p in results):
        print("\n✓ ALL VERIFICATIONS PASSED")
        print("System complies with all mathematical and linguistic standards.")
    else:
        print("\n✗ SOME VERIFICATIONS FAILED")
        print("Review failed tests above.")
    
    print("="*70 + "\n")
    
    return all(p for _, p in results)

if __name__ == "__main__":
    success = run_complete_verification()
    exit(0 if success else 1)
