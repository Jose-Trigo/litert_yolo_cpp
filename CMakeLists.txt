cmake_minimum_required(VERSION 3.10)
project(litert_test CXX)

set(CMAKE_CXX_STANDARD 17)

# LiteRT source root
set(LITERT_SRC "/home/efacec/litert_cc_sdk/litert_cc_sdk")

# LiteRT build root
set(LITERT_BUILD "/home/efacec/litert_cc_sdk/litert_cc_sdk_build")

# LiteRT source includes
include_directories(${LITERT_SRC})
include_directories(${LITERT_SRC}/litert)
include_directories(${LITERT_SRC}/litert/cc)
include_directories(${LITERT_SRC}/litert/c)
include_directories(${LITERT_SRC}/litert/build_common)

# LiteRT *generated* includes (THIS IS THE MISSING ONE)
include_directories(${LITERT_BUILD}/include)

# Abseil includes
include_directories(${LITERT_BUILD}/abseil-cpp)

# Flatbuffers includes
include_directories(${LITERT_BUILD}/flatbuffers/include)

# Library directory
link_directories(${LITERT_BUILD})

add_executable(litert_test ../main.cpp)

target_link_libraries(litert_test
    litert_cc_api
    pthread
    dl
)
