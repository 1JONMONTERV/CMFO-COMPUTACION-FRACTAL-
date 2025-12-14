# Building the C++ Matrix Engine

The high-performance Matrix Engine source code is located in this directory.

## Prerequisites
You need a C++ compiler installed and in your PATH:
- **Windows:** Visual Studio 2022 (MSVC) or MinGW (GCC).
- **Linux/Mac:** GCC or Clang.

## Compilation

Run the build script from the root of the repo:
```cmd
scripts\compile_cpp_matrix.bat
```

Or manually verify with:
```bash
g++ -O3 -std=c++17 -o matrix_test.exe matrix_engine.cpp test_matrix.cpp
./matrix_test.exe
```

## Architecture
- `matrix_engine.hpp`: Defines the 7x7 Complex Matrix class.
- `matrix_engine.cpp`: Implements O(1) Unitary checks and O(N^3) Multiplication.
