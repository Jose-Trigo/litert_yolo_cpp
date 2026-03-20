cmake_minimum_required(VERSION 3.10)
project(litert_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to your LiteRT build
set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3/litert/build-pi-clang")

# Include directories
include_directories(
    ${LITERT_ROOT}
    ${LITERT_ROOT}/runtime
    ${LITERT_ROOT}/c
)

# LiteRT libraries
set(LITERT_LIBS
    ${LITERT_ROOT}/libLiteRt.so
    ${LITERT_ROOT}/runtime/liblitert_runtime.a
    ${LITERT_ROOT}/c/liblitert_runtime_c_api_shared_lib.so
)

# Your benchmark program
add_executable(litert_bench main.cpp)

# Link LiteRT
target_link_libraries(litert_bench
    ${LITERT_LIBS}
    pthread
    dl
)
