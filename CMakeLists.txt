cmake_minimum_required(VERSION 3.10)
project(litert_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3/litert")
set(LITERT_BUILD "${LITERT_ROOT}/build-pi-clang")

# Include directories (ONLY build-pi-clang)
include_directories(
    # C++ API
    ${LITERT_BUILD}/cc/include

    # C API (C++ API depends on these)
    ${LITERT_BUILD}/c/include

    # Generated build_common headers
    ${LITERT_BUILD}/include

    # Abseil (required by LiteRT C++ API)
    ${LITERT_BUILD}/abseil-cpp
    ${LITERT_BUILD}/abseil-cpp/include
)

# Libraries (ONLY build-pi-clang)
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
