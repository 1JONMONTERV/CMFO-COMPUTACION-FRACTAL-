/**
 * TypeScript definitions for @cmfo/core
 */

/**
 * Get the phi constant (golden ratio)
 * @returns The phi constant (≈ 1.618033988749895)
 */
export function phi(): number;

/**
 * Compute T7 tensor product of two 7-element vectors
 * @param a - First 7-element vector
 * @param b - Second 7-element vector
 * @returns Result 7-element vector
 */
export function tensor7(a: number[], b: number[]): number[];

/**
 * Compute gamma step function
 * @param x - Input value
 * @returns Gamma step result
 */
export function gammaStep(x: number): number;

/**
 * T7Tensor class for 7-dimensional tensor operations
 */
export class T7Tensor {
    /**
     * The tensor data (7 elements)
     */
    data: number[];

    /**
     * Create a T7 tensor
     * @param data - 7-element array
     */
    constructor(data: number[]);

    /**
     * Multiply with another T7 tensor
     * @param other - Another tensor or 7-element array
     * @returns Result tensor
     */
    multiply(other: T7Tensor | number[]): T7Tensor;

    /**
     * Get tensor as array
     * @returns The tensor data
     */
    toArray(): number[];

    /**
     * String representation
     * @returns String representation of the tensor
     */
    toString(): string;
}

/**
 * Display CMFO information to console
 */
export function info(): void;
