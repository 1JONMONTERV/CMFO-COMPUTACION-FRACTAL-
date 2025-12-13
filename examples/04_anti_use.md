# 04. What CMFO is NOT for (Anti-Use Cases)

To ensure success, use the right tool for the job. CMFO is specialized.

## ❌ Do NOT use CMFO for:

### 1. Generating Creative Fiction (Chatbot Style)
**Why:** CMFO is a deterministic kernel. It does not "dream" or "hallucinate". If you ask it to write a poem about a toaster, without a specific semantic path, it will return a null tensor or a precise geometric definition, not a rhyming couplet.
**Use:** GPT-4 / Llama / Claude.

### 2. Standard Classification via Softmax
**Why:** CMFO logic does not use probability distributions summing to 1.0. It uses resonance basins. If your downstream app expects `[0.1, 0.2, 0.7]`, CMFO will break it unless you add a specific adapter layer.
**Use:** Scikit-Learn / PyTorch.

### 3. Cryptography (RSA/ECC Replacement)
**Why:** While CMFO has cryptographic properties (fractal hashing), it is **not yet audited** by NIST. Do not use it to secure banking transactions or nuclear codes yet.
**Use:** OpenSSL / Sodium.

## ✅ DO use CMFO for:
- Consistent semantic embedding.
- Deterministic logic flows.
- High-entropy signal compression.
