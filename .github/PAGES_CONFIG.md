# GitHub Pages Configuration

## Current Setup
GitHub Pages should be configured to serve from:
- **Source:** Deploy from a branch
- **Branch:** `main`
- **Folder:** `/docs`

## How to Configure (Manual Steps)
1. Go to: https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-/settings/pages
2. Under "Build and deployment":
   - Source: **Deploy from a branch**
   - Branch: **main**
   - Folder: **/docs**
3. Click **Save**

## Result
The website will be available at:
https://1jonmonterv.github.io/CMFO-COMPUTACION-FRACTAL-/

Any changes to files in the `docs/` folder on the `main` branch will automatically update the website within 1-2 minutes.

## Note
No GitHub Actions workflow is needed. GitHub Pages will automatically rebuild when you push changes to `docs/`.
