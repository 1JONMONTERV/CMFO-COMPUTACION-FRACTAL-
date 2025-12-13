/**
 * @file cmfo_tolerances.h
 * @brief Numerical tolerance constants for CMFO-UNIVERSE
 *
 * This header defines all numerical tolerances used throughout the codebase
 * to ensure consistent precision standards and reproducible results.
 *
 * See Docs/NUMERICAL_TOLERANCES.md for detailed rationale and mathematical justification.
 */

#ifndef CMFO_TOLERANCES_H
#define CMFO_TOLERANCES_H

// ============================================================================
// FUNDAMENTAL CONSTANTS
// ============================================================================

/**
 * @brief Golden Ratio (φ) - Fundamental to φ-logic
 * Precision: 15 decimal digits
 * Verification: φ² = φ + 1 (error < 1e-15)
 */
#define CMFO_PHI 1.618033988749895

// ============================================================================
// MATRIX OPERATIONS (T⁷ Algebra)
// ============================================================================

/**
 * @brief Tolerance for matrix multiplication verification
 * Rationale: Accumulation over 49 operations per element
 */
#define CMFO_TOL_MAT_MUL 1e-14

/**
 * @brief Tolerance for matrix inversion verification (A * inv(A) = I)
 * Rationale: Error propagation through determinant and adjugate computation
 */
#define CMFO_TOL_MAT_INV 1e-12

/**
 * @brief Tolerance for determinant calculation
 * Rationale: 7×7 determinant involves 5040 terms (7! permutations)
 */
#define CMFO_TOL_DET 1e-13

// ============================================================================
// TENSOR OPERATIONS
// ============================================================================

/**
 * @brief Tolerance for tensor product operations
 * Rationale: Element-wise multiplication with minimal error accumulation
 */
#define CMFO_TOL_TENSOR 1e-14

// ============================================================================
// SOLITON SIMULATIONS (Sine-Gordon)
// ============================================================================

/**
 * @brief Tolerance for energy conservation in soliton dynamics
 * Rationale: Finite-difference discretization error O(dx²) ≈ 4e-4
 */
#define CMFO_TOL_SOLITON_ENERGY 1e-6

/**
 * @brief Tolerance for topological charge (winding number)
 * Rationale: Discrete integration of field gradients
 */
#define CMFO_TOL_TOPOLOGICAL 1e-5

/**
 * @brief Tolerance for Lorentz gamma factor invariant
 * Rationale: γ = 1/√(1-v²) must be consistent across transformations
 */
#define CMFO_TOL_GAMMA 1e-12

// ============================================================================
// CUDA KERNEL TOLERANCES
// ============================================================================

/**
 * @brief Tolerance for GPU vs CPU result comparison
 * Rationale: Different arithmetic instruction sequences on GPU
 */
#define CMFO_TOL_GPU_CPU 1e-10

/**
 * @brief Tolerance for theta (phase) evolution in dynamics kernels
 * Rationale: Time-stepping accumulation with trigonometric functions
 */
#define CMFO_TOL_THETA 1e-8

// ============================================================================
// PHI-LOGIC
// ============================================================================

/**
 * @brief Tolerance for φ-logic convergence to boolean logic
 * Rationale: Verifies lim(φ→1) [φ-logic] = boolean logic
 */
#define CMFO_TOL_PHI_LOGIC 1e-6

// ============================================================================
// GENERAL UTILITIES
// ============================================================================

/**
 * @brief Default tolerance for general floating-point comparisons
 */
#define CMFO_TOL_DEFAULT 1e-12

/**
 * @brief Machine epsilon for double precision
 * Note: Typically ~2.22e-16, but defined explicitly for clarity
 */
#define CMFO_EPSILON 2.220446049250313e-16

#endif  // CMFO_TOLERANCES_H
