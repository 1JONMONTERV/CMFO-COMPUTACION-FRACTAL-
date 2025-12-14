# CMFO Publishing Guide

## 1. Publishing to PyPI (Python Package)

Your `ci.yml` is configured to publish automatically when you push a tag like `v1.0.0`.
However, you need to give it the "Secret Key".

### Steps:
1.  **Get Token:** Log in to [pypi.org](https://pypi.org), go to Account Settings -> API Tokens -> Add API Token. Scope it to "Entire account" (for now) or "Project: cmfo".
2.  **Add to GitHub:**
    *   Go to your Repo (`CMFO-COMPUTACION-FRACTAL-`).
    *   Settings -> Secrets and variables -> Actions.
    *   Click **New repository secret**.
    *   Name: `PYPI_TOKEN`
    *   Value: (Paste the token starting with `pypi-...`).
3.  **Trigger Release:**
    ```bash
    git tag v1.0.0
    git push origin v1.0.0
    ```
    The `ci.yml` robot will catch this tag, build the package, and upload it to PyPI.

## 2. Website Deployment

We added a robot (`deploy-web.yml`) that updates the site on every push to `main`.
*   **Status:** Active.
*   **URL:** https://1jonmonterv.github.io/CMFO-COMPUTACION-FRACTAL-/
*   If it doesn't appear, check the "Actions" tab in GitHub to see if the `Deploy to GitHub Pages` job succeeded.

## 3. C++ / CUDA Binaries

To publish compiled binaries (`libcmfo.so` or `cmfo.dll`):
1.  We need to add a "Release" job to `ci.yml` (Advanced).
2.  For now, your repository provides the **Source Code** (`core/native`), which is standard for high-performance libraries (users compile locally for their specific GPU).
