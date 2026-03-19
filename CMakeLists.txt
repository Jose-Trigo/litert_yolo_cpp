cmake_minimum_required(VERSION 3.10)
project(litert_test CXX)

set(CMAKE_CXX_STANDARD 17)

# LiteRT root
set(LITERT_ROOT "/home/efacec/litert_cc_sdk/litert_cc_sdk")

# Include directories (these are the REAL paths on your system)
include_directories(${LITERT_ROOT}/litert)
include_directories(${LITERT_ROOT}/litert/cc)
include_directories(${LITERT_ROOT}/litert/c)

# Library directory
link_directories(/home/efacec/litert_cc_sdk/litert_cc_sdk_build)

add_executable(litert_test ../main.cpp)

target_link_libraries(litert_test
    litert_cc_api
    pthread
    dl
)
