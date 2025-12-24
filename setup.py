
from setuptools import setup, Extension, find_packages
import os

# Define the native extension
native_module = Extension(
    name='cmfo_core_native',
    sources=['src/jit/nvrtc_bridge.cpp'],
    include_dirs=[
        os.environ.get('CUDA_PATH', '') + '/include',
        'src/jit'
    ],
    library_dirs=[
        os.environ.get('CUDA_PATH', '') + '/lib/x64',
    ],
    libraries=['nvrtc', 'cuda'],
    extra_compile_args=['/std:c++17', '/O2'] if os.name == 'nt' else ['-std=c++17', '-O3']
)

setup(
    name='cmfo',
    version='3.1.0',
    description='CMFO: Computational Manifold Fractal Operators',
    packages=find_packages('bindings/python'),
    package_dir={'': 'bindings/python'},
    ext_modules=[native_module] if os.environ.get('CUDA_PATH') else [],
    install_requires=[],
)
