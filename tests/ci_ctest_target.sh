#!/usr/bin/env bash
# Helper script to run CTest and produce JUnit XML output for local/dev use
set -euo pipefail
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
cd build
ctest --output-on-failure --output-junit test-results.xml || true
ls -la
echo "JUnit XML produced at build/test-results.xml"