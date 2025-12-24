#!/bin/bash
# Build script for CMFO native libraries
# This script compiles the C/C++ core libraries needed by Python and Node.js bindings

set -e  # Exit on error

echo "=== CMFO Native Library Build Script ==="
echo ""

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    LIB_EXT="so"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    LIB_EXT="dylib"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="windows"
    LIB_EXT="dll"
else
    echo "Unknown platform: $OSTYPE"
    exit 1
fi

echo "Platform detected: $PLATFORM"
echo ""

# Configuration
BUILD_DIR="core/native/build"
INSTALL_DIR="core/native/lib"
PYTHON_BINDING_DIR="bindings/python/cmfo"
NODE_BINDING_DIR="bindings/node/lib"

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR"
mkdir -p "$PYTHON_BINDING_DIR/core"
mkdir -p "$NODE_BINDING_DIR"

# Build with CMake (if CMakeLists.txt exists)
if [ -f "core/native/CMakeLists.txt" ]; then
    echo "Building with CMake..."
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    cd ../../..
    echo "✓ CMake build complete"
else
    echo "No CMakeLists.txt found, skipping CMake build"
fi

# Copy libraries to binding directories
echo ""
echo "Copying libraries to binding directories..."

if [ -f "$BUILD_DIR/Release/cmfo_core.$LIB_EXT" ]; then
    cp "$BUILD_DIR/Release/cmfo_core.$LIB_EXT" "$PYTHON_BINDING_DIR/core/"
    cp "$BUILD_DIR/Release/cmfo_core.$LIB_EXT" "$NODE_BINDING_DIR/"
    echo "✓ Copied cmfo_core.$LIB_EXT"
elif [ -f "$BUILD_DIR/cmfo_core.$LIB_EXT" ]; then
    cp "$BUILD_DIR/cmfo_core.$LIB_EXT" "$PYTHON_BINDING_DIR/core/"
    cp "$BUILD_DIR/cmfo_core.$LIB_EXT" "$NODE_BINDING_DIR/"
    echo "✓ Copied cmfo_core.$LIB_EXT"
else
    echo "⚠ Warning: cmfo_core.$LIB_EXT not found"
fi

# Build CUDA libraries if available
if command -v nvcc &> /dev/null; then
    echo ""
    echo "CUDA detected, building CUDA kernels..."
    
    if [ -f "core/native/cuda/cmfo_cuda.cu" ]; then
        nvcc -shared -o "$BUILD_DIR/cmfo_cuda.$LIB_EXT" \
            core/native/cuda/cmfo_cuda.cu \
            -O3 -arch=sm_70
        
        cp "$BUILD_DIR/cmfo_cuda.$LIB_EXT" "$PYTHON_BINDING_DIR/core/"
        echo "✓ Built and copied CUDA library"
    else
        echo "⚠ CUDA source files not found"
    fi
else
    echo ""
    echo "CUDA not available, skipping GPU build"
fi

echo ""
echo "=== Build Complete ==="
echo ""
echo "Libraries installed to:"
echo "  - Python: $PYTHON_BINDING_DIR/core/"
echo "  - Node.js: $NODE_BINDING_DIR/"
echo ""
echo "Next steps:"
echo "  1. Test Python: cd bindings/python && python -c 'import cmfo; cmfo.info()'"
echo "  2. Test Node.js: cd bindings/node && npm test"
