# Test discovery and generation macros
MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
    SET(dirlist "")
    FOREACH(child ${children})
        IF(IS_DIRECTORY ${curdir}/${child})
            LIST(APPEND dirlist ${child})
        ENDIF()
    ENDFOREACH()
    SET(${result} ${dirlist})
ENDMACRO()

# Function to add CUDA support to a test target
function(add_cuda_to_test TARGET_NAME)
    if(ENABLE_CUDA)
        set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)
        set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_definitions(${TARGET_NAME} PRIVATE __CUDA_ARCH_LIST__=600)
    endif()
endfunction()

# Function to create a test executable from a header file
function(add_generated_test TARGET_NAME TEST_SOURCE EXTENSION)
    set(TEST_GENERATED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_${EXTENSION}/launcher.${EXTENSION}")
    
    # Create the output directory
    get_filename_component(TEST_OUTPUT_DIR ${TEST_GENERATED_SOURCE} DIRECTORY)
    file(MAKE_DIRECTORY ${TEST_OUTPUT_DIR})
    
    # Configure the launcher file
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/launcher.in ${TEST_GENERATED_SOURCE} @ONLY)
    
    set(TARGET_NAME_EXT "${TARGET_NAME}_${EXTENSION}")
    
    # Create the executable
    add_executable(${TARGET_NAME_EXT} ${TEST_GENERATED_SOURCE})
    target_sources(${TARGET_NAME_EXT} PRIVATE main.cpp)
    
    # Set C++ standard
    set_target_properties(${TARGET_NAME_EXT} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO)
    
    # Include directories
    target_include_directories(${TARGET_NAME_EXT} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR})
    
    # Link FKL library
    target_link_libraries(${TARGET_NAME_EXT} PRIVATE FKL::FKL)
    
    # Handle NVRTC dependencies
    if(NVRTC_ENABLE)
        target_link_libraries(${TARGET_NAME_EXT} PRIVATE ${NVRTC_LIBRARIES})
        target_compile_definitions(${TARGET_NAME_EXT} PRIVATE NVRTC_ENABLED)
        target_link_libraries(${TARGET_NAME_EXT} PRIVATE CUDA::cuda_driver CUDA::cudart)
        
        if(MSVC AND NVRTC_STATIC_LINK)
            set_target_properties(${TARGET_NAME_EXT} PROPERTIES
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        endif()
        
        # Add LLVM JIT support if enabled
        if(NVRTC_JIT_ENABLED)
            target_include_directories(${TARGET_NAME_EXT} PRIVATE ${LLVM_JIT_INCLUDE_DIRS})
            target_compile_options(${TARGET_NAME_EXT} PRIVATE ${LLVM_JIT_CXX_FLAGS})
            target_link_libraries(${TARGET_NAME_EXT} PRIVATE ${LLVM_JIT_LIBRARIES})
            target_compile_definitions(${TARGET_NAME_EXT} PRIVATE LLVM_JIT_ENABLED)
            message(STATUS "Linking LLVM JIT libraries to test: ${TARGET_NAME_EXT}")
        endif()
    endif()
    
    # Handle LLVM JIT support for Clang interpreter tests (independent of NVRTC)
    file(READ ${TEST_SOURCE} TEST_CONTENT_CHECK)
    string(FIND "${TEST_CONTENT_CHECK}" "clang/Interpreter/Interpreter.h" POS_CLANG_INTERPRETER_CHECK)
    if(POS_CLANG_INTERPRETER_CHECK GREATER_EQUAL 0 AND LLVM_JIT_ENABLED)
        target_include_directories(${TARGET_NAME_EXT} PRIVATE ${LLVM_JIT_INCLUDE_DIRS})
        target_compile_options(${TARGET_NAME_EXT} PRIVATE ${LLVM_JIT_CXX_FLAGS})
        target_link_libraries(${TARGET_NAME_EXT} PRIVATE ${LLVM_JIT_LIBRARIES})
        target_compile_definitions(${TARGET_NAME_EXT} PRIVATE LLVM_JIT_ENABLED)
        message(STATUS "Linking LLVM JIT libraries to Clang interpreter test: ${TARGET_NAME_EXT}")
    endif()
    
    # Handle CUDA-specific configuration
    if(${EXTENSION} STREQUAL "cu" AND ENABLE_CUDA)
        add_cuda_to_test(${TARGET_NAME_EXT})
    endif()
    
    # Add compiler-specific flags
    if(MSVC)
        target_compile_options(${TARGET_NAME_EXT} PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/diagnostics:caret>
            $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    endif()
    
    # Add as a test
    add_test(NAME ${TARGET_NAME_EXT} COMMAND ${TARGET_NAME_EXT})
    
    message(STATUS "Added test: ${TARGET_NAME_EXT}")
endfunction()

# Discover and add tests
function(discover_tests)
    file(GLOB_RECURSE TEST_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
    
    foreach(test_source ${TEST_SOURCES})
        get_filename_component(TARGET_NAME ${test_source} NAME_WE)
        file(READ ${test_source} TEST_SOURCE_CONTENTS)
        
        # Check for special markers in the test files
        string(FIND "${TEST_SOURCE_CONTENTS}" "__ONLY_CU__" POS_ONLY_CU)
        string(FIND "${TEST_SOURCE_CONTENTS}" "__ONLY_CPU__" POS_ONLY_CPU)
        string(FIND "${TEST_SOURCE_CONTENTS}" "#if defined(NVRTC_ENABLED)" POS_NVRTC)
        string(FIND "${TEST_SOURCE_CONTENTS}" "clang/Interpreter/Interpreter.h" POS_CLANG_INTERPRETER)
        
        # Add C++ version if not marked as CUDA-only
        if(${POS_ONLY_CU} EQUAL -1)
            # Check if test needs NVRTC and if NVRTC is enabled
            if(POS_NVRTC GREATER_EQUAL 0 AND NOT NVRTC_ENABLE)
                message(STATUS "Skipping NVRTC test ${TARGET_NAME} because NVRTC is disabled")
            # Check if test needs Clang interpreter and if LLVM JIT is enabled
            elseif(POS_CLANG_INTERPRETER GREATER_EQUAL 0 AND NOT LLVM_JIT_ENABLED)
                message(STATUS "Skipping Clang interpreter test ${TARGET_NAME} because LLVM JIT is disabled")
            else()
                add_generated_test("${TARGET_NAME}" "${test_source}" "cpp")
            endif()
        endif()
        
        # Add CUDA version if CUDA is available and not marked as CPU-only
        if(CMAKE_CUDA_COMPILER AND ENABLE_CUDA AND ${POS_ONLY_CPU} EQUAL -1)
            add_generated_test("${TARGET_NAME}" "${test_source}" "cu")
        endif()
    endforeach()
endfunction()

# Discover and build tests
discover_tests()