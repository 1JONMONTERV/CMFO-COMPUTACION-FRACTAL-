from setuptools import setup, find_packages

setup(
    name="cmfo",
    version="1.1.0",
    author="Jonathan Montero Viques",
    author_email="jesuslocopor@gmail.com",
    description="Fractal Universal Computation Engine",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-",
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
