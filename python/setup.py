from setuptools import setup, find_packages

setup(
    name="cmfo-universe",
    version="1.0.0",
    description="Deterministic Fractal Computing Engine based on T7 Geometry and Phi-Logic.",
    author="Jonnathan Montero Víquez",
    author_email="jmvlavacar@hotmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "cmfo-stress=cmfo.stress:main",
            "cmfo-visualize=cmfo.soliton_visualize:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
