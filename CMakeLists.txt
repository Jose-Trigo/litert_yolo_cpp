cmake_minimum_required(VERSION 3.16)
project(litert_c_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LITERT_ROOT  "/home/efacec/LiteRT-2.1.3")
set(LITERT_SRC   "${LITERT_ROOT}/litert")
set(LITERT_BUILD "${LITERT_SRC}/build-pi-clang")

add_executable(litert_c_bench main.cpp)

# IMPORTANT: only the build-exported headers (single API surface)
target_include_directories(litert_c_bench PRIVATE
  "${LITERT_BUILD}/include"
)

target_link_libraries(litert_c_bench PRIVATE
  "${LITERT_BUILD}/c/libLiteRt.so"
  pthread
  dl
)

set_target_properties(litert_c_bench PROPERTIES
  BUILD_RPATH "${LITERT_BUILD}/c"
)