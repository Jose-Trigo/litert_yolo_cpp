cmake_minimum_required(VERSION 3.10)
project(litert_c_benchmark C)

set(CMAKE_C_COMPILER clang)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3")
set(LITERT_SRC "${LITERT_ROOT}/litert")
set(LITERT_BUILD "${LITERT_SRC}/build-pi-clang")

# Include ONLY the C API headers
include_directories(
    ${LITERT_SRC}/c
    ${LITERT_BUILD}/include
)

# Link ONLY the C runtime library
set(LITERT_LIBS
    ${LITERT_BUILD}/c/libLiteRt.so
)

add_executable(litert_c_bench ../main.c)

target_link_libraries(litert_c_bench
    ${LITERT_LIBS}
    pthread
    dl
)
