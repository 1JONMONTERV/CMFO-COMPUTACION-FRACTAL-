@echo off
echo Initialization Visual Studio Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo Building C++ Extension...
cd bindings/python
python setup.py build_ext --inplace

echo.
echo Running Benchmarks...
python ../../benchmarks/benchmark_native.py
python ../../benchmarks/benchmark_simulation.py
python ../../benchmarks/benchmark_superposition.py
