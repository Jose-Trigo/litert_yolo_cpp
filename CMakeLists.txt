cmake_minimum_required(VERSION 3.10)
project(litert_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3/litert")
set(LITERT_BUILD "${LITERT_ROOT}/build-pi-clang")

# Include directories (source tree FIRST)
include_directories(
    ${LITERT_ROOT}
    ${LITERT_ROOT}/c
    ${LITERT_ROOT}/cc
    ${LITERT_ROOT}/runtime
    ${LITERT_ROOT}/build_common

    # build-tree includes AFTER source-tree
    ${LITERT_BUILD}/include
    ${LITERT_BUILD}/cc/include
    ${LITERT_BUILD}/c/include
)

# Libraries
set(LITERT_LIBS
    ${LITERT_BUILD}/c/libLiteRt.so
    ${LITERT_BUILD}/runtime/liblitert_runtime.a
    ${LITERT_BUILD}/cc/liblitert_cc_api.a
)

add_executable(litert_bench ../main.cpp)

target_link_libraries(litert_bench
    ${LITERT_LIBS}
    pthread
    dl
)
