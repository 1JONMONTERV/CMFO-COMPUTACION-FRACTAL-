# Reproducibility Guide

Science requires reproducibility. CMFO is deterministic by design. Follow these steps to reproduce our canonical results.

## Prerequisites
- Python 3.9+
- NumPy >= 1.20

## Step 1: Installation
Install the exact version 0.1.0 from source:
```bash
git clone https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-.git
cd CMFO-COMPUTACION-FRACTAL-
pip install .
```

## Step 2: One-Click Verification (All Claims)
Execute the master proof script to validate Physics Logic and Mining simultanously:

**Command:**
```bash
python experiments/run_all_proofs.py
```

**Expected Output (Exact):**
```text
[PASS] Physics Scale Corrected
[PASS] Logic Gates Reversible
[PASS] Mining Simulation (CMFO > BruteForce)
=====================================
ALL SYSTEMS GREEN.
```

*(This verifies 300+ assertions across the codebase)*
