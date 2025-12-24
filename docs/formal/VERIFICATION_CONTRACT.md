# CMFO Verification Contract
**Date:** December 17, 2025
**Scope:** Mathematical Auditing of CMFO Implementations

This document defines the mandatory mathematical invariants that any CMFO implementation (or discretization) must satisfy to be considered "formally correct". These criteria bridge the gap between the abstract Dirichlet form theory and the concrete discrete automaton.

## I. Mathematical Invariants (Dirichlet Consistency)

For a discrete Laplacian approximation $\Delta_K$ on a mesh/graph of depth $K$:

| ID | Invariant | Description | Tolerance |
| :--- | :--- | :--- | :--- |
| **I1** | **Symmetry** | $\langle \Delta_K f, g \rangle = \langle f, \Delta_K g \rangle$ | $\epsilon < 10^{-14}$ (Double) |
| **I2** | **Positivity** | $\langle -\Delta_K f, f \rangle \ge 0$ (semi-definiteness) | $\ge -10^{-15}$ |
| **I3** | **Kernel** | $\Delta_K \mathbf{1} = 0$ (constants are harmonic) | $\epsilon < 10^{-14}$ |
| **I4** | **Max Principle** | $e^{t\Delta_K} f \ge 0$ if $f \ge 0$ (Markovianity) | Strict (0 tolerance) |
| **I5** | **Spectra** | $\lambda_j(\Delta_K) \to \lambda_j(\Delta)$ (Mosco convergence) | Monotonic conv. check |

## II. Functional Tests (Dynamical Consistency)

For the time-evolution operator $U_\Delta(t)$:

| ID | Test | Description |
| :--- | :--- | :--- |
| **F1** | **Symplecticity** | Energy drift must be bounded by $\mathcal{O}(\Delta t^2)$ over long timescales. |
| **F2** | **Reversibility** | $U^{-1}(\Delta t) \circ U(\Delta t) \mathbf{X}_0 = \mathbf{X}_0$ exactly (or machine precision). |
| **F3** | **Gauge Covariance** | If coupled to $A$, local phase rotations in $A$ must commute with evolution up to gauge transform. |

## Usage
Include this contract in the CI/CD pipeline. Every release build must pass I1-I4 and F1-F2.
