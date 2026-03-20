cmake_minimum_required(VERSION 3.20)
project(litert_yolo_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adjust if needed
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

add_executable(litert_cpu_bench main.cpp)

# Headers for:
# - "litert/cc/..."
# - "absl/types/span.h"
target_include_directories(litert_cpu_bench PRIVATE
  "${LITERT_ROOT}"                               # parent of litert/
  "${LITERT_BUILD}/_deps/com_google_absl-src"   # absl headers
)

# Parse run_model link command and reuse all its link items.
file(READ "${RUN_MODEL_LINK_TXT}" RUN_MODEL_LINK_CMD)
separate_arguments(RUN_MODEL_LINK_ARGS UNIX_COMMAND "${RUN_MODEL_LINK_CMD}")

set(REUSED_LINK_ITEMS "")
foreach(arg IN LISTS RUN_MODEL_LINK_ARGS)
  # Keep libs and linker flags only
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

# Reuse the same link universe as run_model
target_link_libraries(litert_cpu_bench PRIVATE ${REUSED_LINK_ITEMS})

# Runtime .so lookup
set_target_properties(litert_cpu_bench PROPERTIES
  BUILD_RPATH "${LITERT_BUILD}/c;${LITERT_BUILD}/cc"
)