cmake_minimum_required(VERSION 3.10)
project(litert_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to your LiteRT build
set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3/litert")
set(LITERT_BUILD "${LITERT_ROOT}/build-pi-clang")

# Include directories
include_directories(
    ${LITERT_ROOT}
    ${LITERT_ROOT}/runtime
    ${LITERT_ROOT}/c
    ${LITERT_BUILD}
)

# LiteRT libraries
set(LITERT_LIBS
    ${LITERT_BUILD}/runtime/liblitert_runtime.a
    ${LITERT_BUILD}/c/liblitert_runtime_c_api_shared_lib.so
)

add_executable(litert_bench main.cpp)

target_link_libraries(litert_bench
    ${LITERT_LIBS}
    pthread
    dl
)
