cmake_minimum_required(VERSION 3.10)
project(litert_cpp_benchmark CXX)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3")
set(LITERT_SRC  "${LITERT_ROOT}/litert")
set(LITERT_BUILD "${LITERT_SRC}/build-pi-clang")

add_executable(litert_cpp_bench main.cpp)

target_include_directories(litert_cpp_bench PRIVATE
    ${LITERT_ROOT}                # needed for #include "litert/cc/..."
    ${LITERT_BUILD}/cc/include    # generated C++ headers
    ${LITERT_BUILD}/include
)

target_link_libraries(litert_cpp_bench PRIVATE
    ${LITERT_BUILD}/c/libLiteRt.so
    pthread
    dl
)

# So runtime finds libLiteRt.so without extra export
set_target_properties(litert_cpp_bench PROPERTIES
    BUILD_RPATH "${LITERT_BUILD}/c"
)