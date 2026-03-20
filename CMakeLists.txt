cmake_minimum_required(VERSION 3.10)
project(litert_cpp_benchmark CXX)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set LiteRT paths (update these if needed)
set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3")
set(LITERT_SRC "${LITERT_ROOT}/litert")
set(LITERT_BUILD "${LITERT_SRC}/build-pi-clang")

# Include both build and source headers for C++ API
include_directories(
    ${LITERT_SRC}
    ${LITERT_SRC}/cc
    ${LITERT_BUILD}/cc/include
    ${LITERT_BUILD}/include
)

# Link the correct LiteRT runtime library (CPU)
set(LITERT_LIBS
    ${LITERT_BUILD}/c/libLiteRt.so
)

add_executable(litert_cpp_bench main.cpp)

target_link_libraries(litert_cpp_bench
    ${LITERT_LIBS}
    pthread
    dl
)