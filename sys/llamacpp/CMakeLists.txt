cmake_minimum_required(VERSION 3.14)
project(go-llama VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use pkg-config to find llama includes
find_package(PkgConfig REQUIRED)

# PKG_CONFIG_PATH should be set by the Makefile
pkg_check_modules(LLAMA REQUIRED libllama)

message(STATUS "Found libllama via pkg-config:")
message(STATUS "  Include dirs: ${LLAMA_INCLUDE_DIRS}")
message(STATUS "  Cflags: ${LLAMA_CFLAGS}")

# Source files
set(WRAPPER_SOURCES
    batch.cpp
    chat_template.cpp
    context.cpp
    decode.cpp
    embedding.cpp
    error.cpp
    completion.cpp
    grammar.cpp
    gpu.cpp
    init.cpp
    logging.cpp
    metadata.cpp
    model.cpp
    runtime.cpp
    sampler.cpp
    tokenizer.cpp
)

# Header files (for IDE support)
set(WRAPPER_HEADERS
    batch.h
    chat_template.h
    context.h
    decode.h
    embedding.h
    error.h
    completion.h
    grammar.h
    gpu.h
    init.h
    logging.h
    metadata.h
    model.h
    runtime.h
    sampler.h
    tokenizer.h
)

# Create static library
add_library(go-llama STATIC ${WRAPPER_SOURCES} ${WRAPPER_HEADERS})

# Include directories from pkg-config
target_include_directories(go-llama PRIVATE
    ${LLAMA_INCLUDE_DIRS}
)

# Apply any cflags from pkg-config
target_compile_options(go-llama PRIVATE
    ${LLAMA_CFLAGS_OTHER}
)

# Detect available GPU backends based on installed libraries
set(LLAMA_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
if(EXISTS "${LLAMA_LIB_DIR}/libggml-metal.a")
    message(STATUS "Metal backend detected")
    target_compile_definitions(go-llama PRIVATE GGML_USE_METAL)
endif()
if(EXISTS "${LLAMA_LIB_DIR}/libggml-cuda.a")
    message(STATUS "CUDA backend detected")
    target_compile_definitions(go-llama PRIVATE GGML_USE_CUDA)
endif()
if(EXISTS "${LLAMA_LIB_DIR}/libggml-vulkan.a")
    message(STATUS "Vulkan backend detected")
    target_compile_definitions(go-llama PRIVATE GGML_USE_VULKAN)
endif()

# Install target
install(TARGETS go-llama
    ARCHIVE DESTINATION lib
)

install(FILES ${WRAPPER_HEADERS}
    DESTINATION include/go-llama
)

# Generate pkg-config file
set(PKG_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/go-llama.pc")
file(WRITE ${PKG_CONFIG_FILE}
"prefix=${CMAKE_INSTALL_PREFIX}

Name: go-llama
Description: Go bindings wrapper library for llama.cpp
Version: ${PROJECT_VERSION}
Cflags: -I\${prefix}/include -I\${prefix}/include/go-llama
Libs: -L\${prefix}/lib -lgo-llama
"
)

install(FILES ${PKG_CONFIG_FILE}
    DESTINATION lib/pkgconfig
)
