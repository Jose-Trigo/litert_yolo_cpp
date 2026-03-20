cmake_minimum_required(VERSION 3.10)
project(litert_cpp_benchmark CXX)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3")
set(LITERT_SRC  "${LITERT_ROOT}/litert")
set(LITERT_BUILD "${LITERT_SRC}/build-pi-clang")

# Find where absl/types/span.h exists
find_path(ABSL_INCLUDE_DIR
    NAMES absl/types/span.h
    PATHS
      "${LITERT_BUILD}/abseil-cpp"
      "${LITERT_BUILD}/_deps/abseil-cpp-src"
      "${LITERT_ROOT}/third_party/absl"
    NO_DEFAULT_PATH
)
if(NOT ABSL_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find absl/types/span.h")
endif()

add_executable(litert_cpp_bench main.cpp)

target_include_directories(litert_cpp_bench PRIVATE
    ${LITERT_ROOT}
    ${LITERT_BUILD}/cc/include
    ${LITERT_BUILD}/include
    ${ABSL_INCLUDE_DIR}
)

target_link_libraries(litert_cpp_bench PRIVATE
    ${LITERT_BUILD}/c/libLiteRt.so
    pthread
    dl
)

set_target_properties(litert_cpp_bench PROPERTIES
    BUILD_RPATH "${LITERT_BUILD}/c"
)