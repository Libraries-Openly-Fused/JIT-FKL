cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_REV 0)
set(PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_REV})

project(JIT-FKL VERSION ${PROJECT_VERSION} LANGUAGES CXX
    DESCRIPTION "JIT compilation tests for Fused Kernel Library"
    HOMEPAGE_URL "https://github.com/Libraries-Openly-Fused/JIT-FKL")

# Enable C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_TESTS "Build unit tests" ON)
option(NVRTC_STATIC_LINK "Enable static linking for NVRTC" ON)

# Find CUDA (mandatory)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA support enabled")
else()
    message(FATAL_ERROR "CUDA compiler not found. CUDA is mandatory for this project. Please install CUDA Toolkit.")
endif()

# LLVM/Clang JIT compilation support (mandatory)
message(STATUS "Configuring LLVM/Clang JIT compilation support...")
find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-18 llvm-config DOC "llvm-config tool")

if(NOT LLVM_CONFIG_EXECUTABLE)
    message(FATAL_ERROR "llvm-config not found. LLVM/Clang JIT compilation is mandatory for this project. Please install LLVM/Clang 18.")
endif()

execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --version OUTPUT_VARIABLE LLVM_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --prefix OUTPUT_VARIABLE LLVM_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir OUTPUT_VARIABLE LLVM_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags OUTPUT_VARIABLE LLVM_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --ldflags OUTPUT_VARIABLE LLVM_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --libs all OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --system-libs OUTPUT_VARIABLE LLVM_SYSTEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

# Find Clang C++ library for interpreter support
find_library(CLANG_CPP_LIB 
    NAMES clang-cpp.18.1 clang-cpp.18 clang-cpp
    PATHS ${LLVM_LIB_DIR} /usr/lib/x86_64-linux-gnu /usr/lib
    NO_DEFAULT_PATH)
if(CLANG_CPP_LIB)
    set(CLANG_LIBRARIES ${CLANG_CPP_LIB})
    message(STATUS "Found Clang C++ library: ${CLANG_CPP_LIB}")
else()
    # Try direct path to the library
    if(EXISTS "/usr/lib/x86_64-linux-gnu/libclang-cpp.so.18.1")
        set(CLANG_LIBRARIES "/usr/lib/x86_64-linux-gnu/libclang-cpp.so.18.1")
        message(STATUS "Using direct path to Clang C++ library: ${CLANG_LIBRARIES}")
    else()
        message(WARNING "libclang-cpp not found, trying individual Clang libraries")
        set(CLANG_LIBRARIES -lclangInterpreter -lclangFrontend -lclangCodeGen -lclangSema -lclangAnalysis -lclangAST -lclangParse -lclangLex -lclangBasic)
    endif()
endif()

# Add Clang include directory for interpreter headers
set(CLANG_INCLUDE_DIR "${LLVM_PREFIX}/include")
if(NOT EXISTS "${CLANG_INCLUDE_DIR}/clang/Interpreter/Interpreter.h")
    # Try alternative location for Clang headers
    set(CLANG_INCLUDE_DIR "/usr/lib/llvm-18/include")
endif()

# Verify that the Clang interpreter header exists (mandatory)
if(NOT EXISTS "${CLANG_INCLUDE_DIR}/clang/Interpreter/Interpreter.h")
    message(FATAL_ERROR "Clang interpreter headers not found. Please install LLVM/Clang 18 development headers.")
endif()

message(STATUS "LLVM/Clang JIT compilation support is enabled.")
message(STATUS "LLVM version: ${LLVM_VERSION}")
message(STATUS "LLVM include dir: ${LLVM_INCLUDE_DIR}")
message(STATUS "Clang include dir: ${CLANG_INCLUDE_DIR}")

set(LLVM_JIT_LIBRARIES ${LLVM_LD_FLAGS} ${LLVM_LIBS} ${CLANG_LIBRARIES} ${LLVM_SYSTEM_LIBS})
set(LLVM_JIT_INCLUDE_DIRS ${LLVM_INCLUDE_DIR} ${CLANG_INCLUDE_DIR})
set(LLVM_JIT_CXX_FLAGS ${LLVM_CXX_FLAGS})

# Setup NVRTC support (mandatory)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. NVRTC support is mandatory for this project.")
endif()

message(STATUS "NVRTC support is enabled.")
if(NVRTC_STATIC_LINK)
    message(STATUS "Configuring for STATIC NVRTC linkage.")
    set(NVRTC_LIBRARIES CUDA::nvrtc_static)
else()
    message(STATUS "Configuring for DYNAMIC NVRTC linkage.")
    set(NVRTC_LIBRARIES CUDA::nvrtc)
endif()

# Find FKL library (from submodule)
set(FKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fkl")
if(EXISTS "${FKL_DIR}/CMakeLists.txt")
    # Build FKL as a subdirectory but disable its tests
    set(BUILD_UTEST OFF CACHE BOOL "Disable FKL unit tests" FORCE)
    set(ENABLE_BENCHMARK OFF CACHE BOOL "Disable FKL benchmarks" FORCE)
    add_subdirectory(${FKL_DIR})
    message(STATUS "Found FKL library in submodule")
else()
    message(FATAL_ERROR "FKL library not found. Please initialize the git submodule: git submodule update --init --recursive")
endif()

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()