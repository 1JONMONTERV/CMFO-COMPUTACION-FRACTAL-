"""
CMFO Python SDK - Setup
"""
from setuptools import setup, find_packages

setup(
    name="cmfo",
    version="1.0.0",
    description="CMFO - Computación Fractal Orientada a Objetos",
    long_description=open("../../README_UNIVERSAL.md").read(),
    long_description_content_type="text/markdown",
    author="CMFO Development Team",
    author_email="info@cmfo.org",
    url="https://github.com/cmfo/cmfo",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="cmfo fractal algebra geometry computation",
    project_urls={
        "Documentation": "https://cmfo.org/docs",
        "Source": "https://github.com/cmfo/cmfo",
        "Tracker": "https://github.com/cmfo/cmfo/issues",
    },
)
