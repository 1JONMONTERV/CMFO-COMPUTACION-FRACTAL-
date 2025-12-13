from setuptools import setup, find_packages

setup(
    name="cmfo-compute",
    version="1.0.2",
    author="Jonnathan Montero",
    author_email="jesuslocopor@gmail.com",
    description="CMFO Compute: Librería base de cálculo fractal T7 del sistema CMFO.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/1JONMONTERV/cmfo-compute",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
