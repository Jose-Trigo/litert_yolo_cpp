cmake_minimum_required(VERSION 3.10)
project(litert_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------
# 1. Set paths to your LiteRT SDK
# ---------------------------------------------------------
set(LITERT_SDK_SRC "/home/efacec/litert_cc_sdk")
set(LITERT_SDK_BUILD "/home/efacec/litert_cc_sdk/litert_cc_sdk_build")

# ---------------------------------------------------------
# 2. Include headers
# ---------------------------------------------------------
include_directories(${LITERT_SDK_SRC})
include_directories(${LITERT_SDK_BUILD}/include)

# ---------------------------------------------------------
# 3. Link the LiteRT static library
# ---------------------------------------------------------
link_directories(${LITERT_SDK_BUILD})

# ---------------------------------------------------------
# 4. Build your test program
# ---------------------------------------------------------
add_executable(litert_test main.cpp)

# ---------------------------------------------------------
# 5. Link libraries
# ---------------------------------------------------------
target_link_libraries(litert_test
    litert_cc_api
    pthread
    dl
)
