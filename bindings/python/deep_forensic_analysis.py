
import struct
import numpy as np
import random
import scipy.stats

def int_to_bits(n):
    return [int(x) for x in f"{n:032b}"]

def int_to_ternary(n):
    # Balanced ternary approximation or simple base 3
    if n == 0: return [0]
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(r)
    return nums[::-1]

def matrix_rank_metric(n):
    # 32 bits -> 4x8 matrix? Or 5x6?
    # Let's try 4x8
    bits = int_to_bits(n)
    mat = np.array(bits).reshape(4, 8)
    return np.linalg.matrix_rank(mat)

def hamming_weight(n):
    return bin(n).count('1')

def longest_run(n):
    s = f"{n:032b}"
    return max(len(c) for c in s.split('0'))

def modulo_profile(n):
    # Check residus
    return [n % 3, n % 5, n % 7, n % 11, n % 13]

def bit_autocorrelation(n):
    bits = np.array(int_to_bits(n))
    # simple lag-1
    if len(bits) < 2: return 0
    return np.corrcoef(bits[:-1], bits[1:])[0,1]

def fractal_smoothness(n):
    # Diff of bits
    bits = np.array(int_to_bits(n))
    return np.sum(np.abs(np.diff(bits)))

class DeepForensics:
    def __init__(self, real_nonce):
        self.real = real_nonce
        self.randoms = [random.randint(0, 2**32-1) for _ in range(2000)]
        
    def analyze(self):
        print(f"Analyzing Real Nonce: {self.real}")
        print(f"Comparing against {len(self.randoms)} random samples...")
        print("-" * 60)
        print(f"{'METRIC':<25} | {'REAL':<10} | {'RAND AVG':<10} | {'SIGMA':<10}")
        print("-" * 60)
        
        metrics = {
            "Hamming Weight": hamming_weight,
            "Longest Run (1s)": longest_run,
            "Matrix Rank (4x8)": matrix_rank_metric,
            "Bit Autocorr (Lag1)": bit_autocorrelation,
            "Fractal Smoothness": fractal_smoothness,
            "Modulo 3": lambda x: x % 3,
            "Modulo 7": lambda x: x % 7,
            "Modulo 31": lambda x: x % 31,
            "Ternary Digit Sum": lambda x: sum(int_to_ternary(x)),
            "XOR Sum (Bytes)": lambda x: sum(bytes(struct.pack("<I", x))),
        }
        
        significance_found = False
        
        for name, func in metrics.items():
            val_real = func(self.real)
            val_rands = [func(r) for r in self.randoms]
            
            # Filter NaNs if any
            val_rands = [v for v in val_rands if not np.isnan(v)]
            if np.isnan(val_real): val_real = 0
            
            mu = np.mean(val_rands)
            sigma = np.std(val_rands)
            
            z_score = 0
            if sigma > 0:
                z_score = abs(val_real - mu) / sigma
                
            print(f"{name:<25} | {val_real:<10.4f} | {mu:<10.4f} | {z_score:<10.2f}")
            
            if z_score > 3.0:
                 print(f"*** ANOMALY DETECTED: {name} ***")
                 significance_found = True
                 
        print("-" * 60)
        if significance_found:
            print("CONCLUSION: Statistical anomalies found. Potential attack vectors identified.")
        else:
            print("CONCLUSION: Real Nonce is statistically indistinguishable from noise in these bases.")

if __name__ == "__main__":
    # Block 905561 Real Nonce
    # 3536931971
    analyzer = DeepForensics(3536931971)
    analyzer.analyze()
