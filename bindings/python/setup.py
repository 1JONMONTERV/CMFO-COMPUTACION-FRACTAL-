import os
from setuptools import setup, find_packages, Extension

# Robustly find the README in the root directory
here = os.path.abspath(os.path.dirname(__file__))
# Point to the PyPI-specific README
local_readme = os.path.join(here, "README_PyPI.md")

try:
    with open(local_readme, encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "CMFO: Continuous Modal Fractal Oscillation Engine (Experimental)"

setup(
    name="cmfo",
    version="0.1.3",
    author="Jonathan Montero Viques",
    author_email="jesuslocopor@gmail.com",
    description="Experimental framework for deterministic fractal computation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
    ],
    ext_modules=[
        Extension(
            "cmfo_core_native",
            sources=["../../core/language/matrix_engine.cpp"],
            include_dirs=["../../core/language"],
            language="c++",
            extra_compile_args=["/std:c++17"] if os.name == 'nt' else ["-std=c++17"]
        )
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-/tree/main/docs",
        "Source": "https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-",
    },
    entry_points={
        "console_scripts": [
            "cmfo=cmfo.cli:main",
        ],
    },
)
