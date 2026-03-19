cmake_minimum_required(VERSION 3.10)
project(litert_c_api_test C)

set(LITERT_BUILD "/home/efacec/litert_cc_sdk/litert_cc_sdk_build")

include_directories(${LITERT_BUILD}/include)
include_directories(${LITERT_BUILD}/abseil-cpp)
include_directories(${LITERT_BUILD}/flatbuffers/include)

link_directories(${LITERT_BUILD})

add_executable(litert_test main.c)

target_link_libraries(litert_test
    litert_c_api
    pthread
    dl
)
