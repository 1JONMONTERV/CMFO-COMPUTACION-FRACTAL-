# CMFO Metrics Specification

## Distance Functions

### 1. Phi-Weighted Euclidean Distance

```
d_φ(x,y) = ||x - y||_φ = √(Σᵢ φⁱ · (xᵢ - yᵢ)²)
```

**Properties:**
- Metric: d_φ(x,y) ≥ 0, d_φ(x,x) = 0
- Symmetry: d_φ(x,y) = d_φ(y,x)
- Triangle inequality: d_φ(x,z) ≤ d_φ(x,y) + d_φ(y,z)

### 2. Angular Distance (Cosine-based)

```
d_angle(x,y) = 1 - cos(θ) = 1 - (⟨x,y⟩_φ / (||x||_φ · ||y||_φ))
```

**Properties:**
- Range: [0, 2]
- Invariant to scaling
- Captures semantic direction, not magnitude

**Recommended:** Use d_angle for semantic similarity, d_φ for geometric proximity.

## Regional Equivalence

### Definition

```
x ~_ε y  ⟺  d_φ(x,y) < ε
```

**Equivalence Relation:**
- Reflexive: x ~_ε x (always, since d_φ(x,x) = 0)
- Symmetric: x ~_ε y ⟹ y ~_ε x
- NOT transitive (approximate equivalence)

### Threshold Values

| Context | ε | Interpretation |
|---------|---|----------------|
| Strict identity | 0.01 | Numerical precision |
| Semantic equivalence | 0.1 | Same concept, minor variation |
| Broad similarity | 0.3 | Related concepts |
| Attractor basin | 0.15 | Default for convergence detection |

## Attractor Definition

### Operational Definition

An **attractor** A_i is a representative vector a_i ∈ X such that:

```
Basin(A_i) = {x ∈ X : d_φ(x, a_i) < ε_basin}
```

Where ε_basin = 0.15 (default).

### Attractor Detection Algorithm

```python
def find_attractor(x, attractors, ε=0.15):
    """
    Returns: (attractor_index, distance) or (None, None)
    """
    for i, a_i in enumerate(attractors):
        dist = d_φ(x, a_i)
        if dist < ε:
            return (i, dist)
    return (None, None)

def add_or_merge(x, attractors, ε=0.15):
    """
    Either merges x into existing attractor or creates new one.
    """
    idx, dist = find_attractor(x, attractors, ε)
    if idx is not None:
        # Convergent: update attractor (moving average)
        attractors[idx] = (attractors[idx] + x) / 2
        return ('convergent', idx)
    else:
        # New attractor
        attractors.append(x)
        return ('new_attractor', len(attractors)-1)
```

## Convergence Metrics

### Convergence Rate

```
Conv_ε(Φ, S) = (# trajectories in existing basins) / (# total trajectories)
```

**Interpretation:**
- Conv < 20%: Highly divergent (chaotic)
- 20% ≤ Conv < 40%: Weakly structured
- 40% ≤ Conv < 70%: Well-structured (target)
- Conv ≥ 70%: Over-convergent (collapsing)

### Attractor Diversity

```
Diversity = (# unique attractors) / (# trajectories)
```

**Interpretation:**
- Diversity ≈ 1.0: Every trajectory unique (divergent)
- Diversity ≈ 0.5: Moderate structure
- Diversity < 0.3: Strong convergence
- Diversity < 0.1: Degenerate (trivial)

### Basin Size Distribution

For each attractor A_i:
```
Size(A_i) = |Basin(A_i)|
```

**Healthy distribution:**
- Power law: few large basins, many small ones
- Largest basin < 30% of total
- No single basin dominates

## Trajectory Analysis

### Trajectory Length

```
L(trajectory) = Σₜ d_φ(x_t, x_{t+1})
```

### Trajectory Stability

```
Stability = 1 - (d_φ(x_final, x_initial) / L(trajectory))
```

**Interpretation:**
- Stability ≈ 1: Circular/loop
- Stability ≈ 0: Straight divergence
- 0.3 < Stability < 0.7: Healthy exploration

## Implementation Notes

**CRITICAL:** Never use hash-based equality (MD5, SHA) for continuous spaces.

**Always use:**
- d_φ or d_angle for distance
- ~_ε for equivalence
- Attractor detection for convergence

**Precision:**
- Store vectors as float64
- Compare with ε ≥ 0.01 (never exact equality)
