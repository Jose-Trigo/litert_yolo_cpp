cmake_minimum_required(VERSION 3.10)
project(litert_cpp_benchmark CXX)

# Set compiler and standard
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set LiteRT paths (update these if needed)
set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3")
set(LITERT_SRC "${LITERT_ROOT}/litert")
set(LITERT_BUILD "${LITERT_SRC}/build-pi-clang")

# Include ONLY the C++ API headers
include_directories(
    ${LITERT_SRC}/cc
    ${LITERT_BUILD}/include
)

# Link ONLY the C++ runtime library
set(LITERT_LIBS
    ${LITERT_BUILD}/cc/libLiteRt.so
)

# Add your minimal C++ benchmark source file
add_executable(litert_cpp_bench minimal_run_model.cc)

target_link_libraries(litert_cpp_bench
    ${LITERT_LIBS}
    pthread
    dl
)