# Contributing to CMFO

Thank you for your interest in contributing to the **Continuous Modal Fractal Oscillation (CMFO)** project.

As a **Scientific Standard**, this repository maintains strict quality controls. We are building the substrate for Aerospace and Financial systems; therefore, "good enough" is not acceptable.

## 🔬 Scientific Rigor

All contributions must adhere to the **Axioms of Geometric Determinism**:
1.  **Bit-Exactness**: New logic must produce identical results on `x86_64` and `ARM64`.
2.  **No Stochasticity**: Do not introduce `random()` calls in the core kernel.
3.  **Formal Proof**: Major algorithmic changes must be accompanied by a LaTeX proof in `docs/theory/`.

## 🛠️ Development Workflow

1.  **Fork & Clone**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/CMFO-COMPUTACION-FRACTAL-.git
    ```
2.  **Install Native Dependencies**:
    You must have a C11 compiler (GCC/Clang) and CMake installed.
3.  **Run Tests**:
    ```bash
    python -m pytest tests/ --verbose
    ```

## 📝 Pull Request Standards

*   **Signed Commits**: All commits must be GPG signed (`git commit -S`).
*   **Linear History**: Rebase before merging. No merge commits.
*   **Documentation**: Update `README.md` and `docs/` if you change public APIs.

## ⚖️ License

By contributing, you agree that your code will be licensed under the **MIT License** held by *Jonathan Montero Viques*.
