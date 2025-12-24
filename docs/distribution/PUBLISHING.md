# Publishing CMFO Packages

This guide explains how to publish CMFO packages to PyPI (Python) and npm (Node.js).

## Prerequisites

### For PyPI (Python)
1. Create an account on [PyPI](https://pypi.org/)
2. Generate an API token in your PyPI account settings
3. Add the token as a GitHub secret named `PYPI_API_TOKEN`

### For npm (Node.js)
1. Create an account on [npmjs.com](https://www.npmjs.com/)
2. Generate an access token: `npm login` then `npm token create`
3. Add the token as a GitHub secret named `NPM_TOKEN`

## Automated Publishing (Recommended)

### Python Package

Publishing is automated via GitHub Actions when you push a version tag:

```bash
# Update version in pyproject.toml and setup.py
# Commit changes
git add bindings/python/pyproject.toml bindings/python/setup.py
git commit -m "Bump version to 1.1.2"

# Create and push tag
git tag v1.1.2
git push origin v1.1.2
```

The workflow will:
- Build the package for multiple Python versions
- Run tests
- Publish to PyPI automatically

### Node.js Package

Publishing is automated via GitHub Actions when you push a node version tag:

```bash
# Update version in package.json
cd bindings/node
npm version patch  # or minor, or major

# Commit and push
git add package.json
git commit -m "Bump node package version"

# Create and push tag
git tag node-v1.0.1
git push origin node-v1.0.1
```

## Manual Publishing

### Python Package

```bash
cd bindings/python

# Install build tools
pip install build twine

# Build the package
python -m build

# Check the package
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

### Node.js Package

```bash
cd bindings/node

# Login to npm (first time only)
npm login

# Publish
npm publish --access public
```

## Version Management

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version: Incompatible API changes
- **MINOR** version: New functionality (backwards-compatible)
- **PATCH** version: Bug fixes (backwards-compatible)

### Python Versioning

Update version in **both** files:
- `bindings/python/pyproject.toml` (line 7)
- `bindings/python/setup.py` (line 16)

### Node.js Versioning

Update version in:
- `bindings/node/package.json`

Or use npm:
```bash
npm version patch  # 1.0.0 → 1.0.1
npm version minor  # 1.0.0 → 1.1.0
npm version major  # 1.0.0 → 2.0.0
```

## Pre-Publication Checklist

### Python Package

- [ ] Version updated in `pyproject.toml` and `setup.py`
- [ ] CHANGELOG updated
- [ ] Tests passing: `pytest tests/`
- [ ] Package builds: `python -m build`
- [ ] Package validates: `twine check dist/*`
- [ ] README.md is up to date
- [ ] LICENSE.txt is included

### Node.js Package

- [ ] Version updated in `package.json`
- [ ] CHANGELOG updated
- [ ] Tests passing: `npm test` (if native lib available)
- [ ] TypeScript definitions are correct
- [ ] README.md is up to date
- [ ] LICENSE file is included

## Testing Before Publishing

### Python

```bash
# Build the package
cd bindings/python
python -m build

# Install locally
pip install dist/cmfo-1.1.1-*.whl

# Test import
python -c "import cmfo; cmfo.info()"

# Uninstall
pip uninstall cmfo
```

### Node.js

```bash
# Install dependencies
cd bindings/node
npm install

# Test locally
npm test

# Or test in another project
cd /path/to/test/project
npm install /path/to/CMFO_GPU_FINAL/bindings/node
```

## Publishing to Test Repositories

### TestPyPI (Python)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ cmfo
```

### npm (Node.js)

npm doesn't have a test registry, but you can:

```bash
# Dry run (shows what would be published)
npm publish --dry-run

# Or use verdaccio (local npm registry)
npm install -g verdaccio
verdaccio
npm publish --registry http://localhost:4873
```

## Troubleshooting

### Python: "File already exists"

PyPI doesn't allow re-uploading the same version. You must bump the version number.

### Node.js: "403 Forbidden"

Check that:
1. You're logged in: `npm whoami`
2. Your token has publish permissions
3. The package name isn't already taken (try a scoped name like `@yourusername/cmfo`)

### Missing Native Libraries

Both packages require the native C/C++ libraries to be built. See the main README for build instructions.

## Post-Publication

After publishing:

1. Create a GitHub release with the same tag
2. Update the main README.md with new version numbers
3. Announce on relevant channels
4. Monitor for issues

## Support

For questions about publishing:
- Email: jmvlavacar@hotmail.com
- GitHub Issues: https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-/issues
