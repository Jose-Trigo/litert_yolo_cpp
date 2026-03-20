cmake_minimum_required(VERSION 3.10)
project(litert_benchmark)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3/litert")
set(LITERT_BUILD "${LITERT_ROOT}/build-pi-clang")

include_directories(
    ${LITERT_ROOT}/c
    ${LITERT_BUILD}
    ${LITERT_BUILD}/cc/include
    ${LITERT_BUILD}/include
    ${LITERT_BUILD}/abseil-cpp
)

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
