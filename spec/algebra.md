# CMFO Algebra Specification

## Formal Definition

### Grammar Algebra G

```
G = (X, ⟨·,·⟩_φ, Γ_φ, R, game, Ω, τ, ⇒)
```

**Components:**

1. **X ⊂ ℝ⁷** - Semantic space (T⁷_φ manifold)
   - Each point represents a linguistic concept
   - Dimension 7 is fundamental (not arbitrary)
   - Complex extension: X ⊂ ℂ⁷ for phase information

2. **⟨·,·⟩_φ : X × X → ℝ** - Phi-weighted inner product
   ```
   ⟨x,y⟩_φ = Σᵢ φⁱ · xᵢ · yᵢ
   where φ = 1.6180339887... (golden ratio)
   ```

3. **Γ_φ : X → X** - Normalization operator
   ```
   Γ_φ(x) = x / ||x||_φ
   where ||x||_φ = √⟨x,x⟩_φ
   ```

4. **R** - Rotation family (grammatical transformations)
   - NEG: Negation (π rotation in semantic plane)
   - TNS: Tense (temporal rotation)
   - MODAL: Modality (epistemic rotation)
   - ASPECT: Aspectual shift
   - VOICE: Active ↔ Passive

5. **game(x,y;θ) : X × X × [0,2π] → X** - Composition operator
   ```
   game(x,y;θ) = Γ_φ(cos(θ)·x + sin(θ)·y)
   ```
   Geometric mixing with angle θ

6. **Ω** - Operator dictionary
   ```
   Ω = {
     CONC_gen: X_N × X_A → X_A  (gender concordance)
     CONC_num: X_N × X_A → X_A  (number concordance)
     APP_s: X_V × X_N → X_VP    (subject application)
     APP_o: X_V × X_N → X_VP    (object application)
     TEMP: X_V × T → X_V        (temporal operator)
     ...
   }
   ```

7. **τ : X → Type** - Type function
   ```
   Type = {N, V, A, Adv, Prep, Det, Pron, ...}
   τ(x) determines grammatical category
   ```

8. **⇒ : Expr → Expr** - Reduction rules (parser)
   ```
   [V x_v] [N x_s] ⇒ APP_s(x_v, x_s)
   [N x_n] [A x_a] ⇒ CONC(x_n, x_a)
   ...
   ```

## Fundamental Constants

- **φ = 1.6180339887...** (Golden Ratio)
- **Dimension = 7** (Manifold dimensionality)
- **ε_norm = 0.01** (Norm tolerance)
- **ε_conv = 0.1** (Convergence threshold)

## Type System

```
N  : Noun
V  : Verb
A  : Adjective
Adv: Adverb
Det: Determiner
Prep: Preposition
Pron: Pronoun

NP : Noun Phrase
VP : Verb Phrase
AP : Adjective Phrase
PP : Prepositional Phrase
S  : Sentence
```

## Operator Signatures

```
CONC_gen : N × A → A
CONC_num : N × A → A
APP_s    : V × N → VP
APP_o    : VP × N → VP
TEMP_t   : V × Tense → V
MOD_n    : N × A → N
DET      : Det × N → NP
PREP     : Prep × NP → PP
```

## Reduction Rules

```
Rule 1 (Subject Application):
[V v] [N s] ⇒ [VP APP_s(v,s)]

Rule 2 (Object Application):
[VP vp] [N o] ⇒ [VP APP_o(vp,o)]

Rule 3 (Modification):
[N n] [A a] ⇒ [N MOD_n(CONC(n,a), n)]

Rule 4 (Determination):
[Det d] [N n] ⇒ [NP DET(d,n)]
```

## Invariants (See laws.md)

All operators must preserve:
1. Closure: op(x,y) ∈ X
2. Norm: ||op(x,y)||_φ ≈ 1
3. Type safety: τ(op(x,y)) = expected_type
