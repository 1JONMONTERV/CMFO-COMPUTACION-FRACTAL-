# Reproducibility Guide

Science requires reproducibility. CMFO is deterministic by design. Follow these steps to reproduce our canonical results.

## Prerequisites
- Python 3.9+
- NumPy >= 1.20

## Step 1: Installation
Install the exact version 0.1.0 from source:
```bash
git clone https://github.com/1JONMONTERV/cmfo-compute.git
cd cmfo-compute
pip install .
```

## Step 2: Canonical Tensor7 Test
Execute the `tensor7` operator with the Golden Vector inputs (See `data/golden_tensor7.json`).

**Command:**
```bash
cmfo tensor7 1.0 0.5
```

**Expected Output (Exact):**
```text
0.8090169943749475
```
*(Note: This is $\varphi/2$ exactly)*

## Step 3: Run the Verification Suite
To validate the entire logic core, including Boolean absorption proofs:

```bash
python -m pytest tests/
```

**Expected Result:**
All tests passed (Green).

## Hash Verification
MD5 of the core logic file (`cmfo/core/api.py` v0.1.0):
*(You can verify this against the released binary)*
