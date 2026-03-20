cmake_minimum_required(VERSION 3.20)


# --- Force Clang (must be before project()) ---
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
# ----------------------------------------------
project(litert_yolo_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3")
set(LITERT_SRC  "${LITERT_ROOT}/litert")
set(LITERT_BUILD "${LITERT_SRC}/build-pi-clang")
set(RUN_MODEL_LINK_TXT "${LITERT_BUILD}/tools/CMakeFiles/run_model.dir/link.txt")

if(NOT EXISTS "${RUN_MODEL_LINK_TXT}")
  message(FATAL_ERROR
    "Missing ${RUN_MODEL_LINK_TXT}\n"
    "Build run_model first:\n"
    "  cd ${LITERT_SRC} && cmake --build ${LITERT_BUILD} --target run_model -j")
endif()

# Abseil headers
set(ABSL_INCLUDE_DIR "${LITERT_BUILD}/abseil-cpp")
if(NOT EXISTS "${ABSL_INCLUDE_DIR}/absl/types/span.h")
  set(ABSL_INCLUDE_DIR "${LITERT_BUILD}/tflite_build/abseil-cpp")
endif()
if(NOT EXISTS "${ABSL_INCLUDE_DIR}/absl/types/span.h")
  message(FATAL_ERROR "Could not find absl/types/span.h")
endif()

# Generated LiteRT headers (contains litert/build_common/build_config.h)
if(NOT EXISTS "${LITERT_BUILD}/include/litert/build_common/build_config.h")
  message(FATAL_ERROR "Missing ${LITERT_BUILD}/include/litert/build_common/build_config.h")
endif()

add_executable(litert_cpu_bench main.cpp)

target_include_directories(litert_cpu_bench PRIVATE
  "${LITERT_ROOT}"           # litert/cc/...
  "${LITERT_BUILD}/include"  # generated headers (build_config.h)
  "${ABSL_INCLUDE_DIR}"      # absl/types/span.h
)

file(READ "${RUN_MODEL_LINK_TXT}" RUN_MODEL_LINK_CMD)
separate_arguments(RUN_MODEL_LINK_ARGS UNIX_COMMAND "${RUN_MODEL_LINK_CMD}")

set(REUSED_LINK_ITEMS "")
foreach(arg IN LISTS RUN_MODEL_LINK_ARGS)
  if(arg MATCHES "\\.(a|so)(\\.[0-9]+)*$"
     OR arg MATCHES "^-l.+"
     OR arg STREQUAL "-pthread"
     OR arg MATCHES "^-Wl,.+")
    list(APPEND REUSED_LINK_ITEMS "${arg}")
  endif()
endforeach()

if(REUSED_LINK_ITEMS STREQUAL "")
  message(FATAL_ERROR "Could not extract link items from ${RUN_MODEL_LINK_TXT}")
endif()

target_link_libraries(litert_cpu_bench PRIVATE ${REUSED_LINK_ITEMS})

set_target_properties(litert_cpu_bench PROPERTIES
  BUILD_RPATH "${LITERT_BUILD}/c;${LITERT_BUILD}/cc"
)