# 01. Installation Guide

This guide ensures you have a clean, working installation of CMFO before attempting complex tasks.

## 1. Install from Source
Run the following from the root of the repository:

```bash
pip install .
```

## 2. Verify CLI
Check that the command-line tool is exposed to your path:

```bash
cmfo --help
```

**Expected Output:**
```text
usage: cmfo [-h] {tensor7} ...

CMFO – Fractal Universal Computation Engine
...
```

## 3. Verify Import
Check that Python can see the library:

```bash
python -c "import cmfo; print('CMFO Installed Successfully')"
```
