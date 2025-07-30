# JIT-FKL Test CMake Documentation

This document explains how to use the CMake build system to compile and run the JIT-FKL tests.

## Prerequisites

### Required Dependencies
- CMake 3.22 or higher
- C++17 compatible compiler (GCC, Clang, MSVC)
- Git (for submodule initialization)

### Optional Dependencies
- **LLVM/Clang 18** (required for `basic_clang_interpreter_test`)
  - On Ubuntu: `sudo apt install llvm-18-dev clang-18 libclang-18-dev`
  - On other systems: Install LLVM/Clang 18 from [LLVM releases](https://github.com/llvm/llvm-project/releases)

- **CUDA Toolkit** (required for `test_nvrtc`)
  - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  - Requires compatible NVIDIA GPU

## Build Instructions

### 1. Initialize Submodules
```bash
git submodule update --init --recursive
```

### 2. Configure the Build
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

#### CMake Options
- `BUILD_TESTS=ON/OFF` - Enable/disable test building (default: ON)
- `NVRTC_ENABLE=ON/OFF` - Enable/disable NVRTC support (default: ON when CUDA is available)
- `NVRTC_STATIC_LINK=ON/OFF` - Use static/dynamic NVRTC linking (default: ON)
- `ENABLE_CUDA=ON/OFF` - Enable/disable CUDA support (default: ON when CUDA compiler is found)

### 3. Build the Tests
```bash
make -j$(nproc)
# or
cmake --build . --parallel
```

### 4. Run the Tests
```bash
# Run all tests
ctest --verbose

# Run a specific test
./bin/basic_clang_interpreter_test_cpp
./bin/test_nvrtc_cpp  # (if NVRTC is enabled)
./bin/test_nvrtc_cu   # (if CUDA is enabled)
```

## Test Structure

The CMake system automatically discovers test header files in the `test/` directory and generates corresponding executables:

### Test Discovery
- All `.h` files in `test/` are scanned for test functions
- Each test must contain a `launch()` function that returns an int (0 = success)
- Tests are automatically categorized based on content and build multiple variants:
  - `*_cpp` - C++ version (always built unless marked `__ONLY_CU__`)
  - `*_cu` - CUDA version (built when CUDA is available unless marked `__ONLY_CPU__`)

### Test Markers
Tests can use special markers to control compilation:
- `__ONLY_CPU__` - Only build C++ version
- `__ONLY_CU__` - Only build CUDA version
- `#if defined(NVRTC_ENABLED)` - Requires NVRTC support
- `#include "clang/Interpreter/Interpreter.h"` - Requires LLVM/Clang JIT support

### Current Tests
1. **basic_clang_interpreter_test** - Tests C++ JIT compilation using Clang interpreter
   - Requires: LLVM/Clang 18
   - Tests: Runtime C++ code compilation and execution

2. **test_nvrtc** - Tests CUDA kernel JIT compilation using NVRTC
   - Requires: CUDA Toolkit, NVRTC, FKL library
   - Tests: Runtime CUDA kernel compilation and execution

## File Structure

```
CMakeLists.txt              # Root CMake configuration
test/
├── CMakeLists.txt          # Test discovery and build configuration
├── launcher.in             # Template for generated launcher files
├── main.cpp                # Main entry point for test executables
├── basic_clang_interpreter_test.h  # Clang interpreter JIT test
└── test_nvrtc.h            # NVRTC CUDA kernel JIT test
fkl/                        # FKL library submodule
build/                      # Build directory (excluded from git)
```

## Dependencies Management

The CMake system automatically handles:
- FKL library compilation from submodule
- LLVM/Clang library detection and linking
- NVRTC library detection and linking
- CUDA compilation flags and architecture settings
- Cross-platform compatibility (Linux, Windows, macOS)

## Troubleshooting

### LLVM/Clang Issues
```bash
# Check if llvm-config is available
which llvm-config-18 || which llvm-config

# Verify Clang headers
ls /usr/lib/llvm-18/include/clang/Interpreter/Interpreter.h
```

### CUDA Issues
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify NVRTC availability
find /usr/local/cuda -name "*nvrtc*"
```

### Build Issues
```bash
# Clean build
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug  # Use Debug for more verbose output
```