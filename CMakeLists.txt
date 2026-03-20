cmake_minimum_required(VERSION 3.20)

# Force clang if not already chosen (works on fresh build dir)
if(NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER clang CACHE STRING "" FORCE)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER clang++ CACHE STRING "" FORCE)
endif()

project(litert_yolo_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LITERT_ROOT "/home/efacec/LiteRT-2.1.3")
set(LITERT_SRC "${LITERT_ROOT}/litert")
set(LITERT_BUILD "${LITERT_SRC}/build-pi-clang")
set(RUN_MODEL_LINK_TXT "${LITERT_BUILD}/tools/CMakeFiles/run_model.dir/link.txt")

if(NOT EXISTS "${RUN_MODEL_LINK_TXT}")
  message(FATAL_ERROR
    "Missing ${RUN_MODEL_LINK_TXT}\n"
    "Build it first:\n"
    "  cd ${LITERT_SRC} && cmake --build ${LITERT_BUILD} --target run_model -j")
endif()

# Abseil include root (pick one that exists)
set(ABSL_INCLUDE_DIR "${LITERT_BUILD}/abseil-cpp")
if(NOT EXISTS "${ABSL_INCLUDE_DIR}/absl/types/span.h")
  set(ABSL_INCLUDE_DIR "${LITERT_BUILD}/tflite_build/abseil-cpp")
endif()
if(NOT EXISTS "${ABSL_INCLUDE_DIR}/absl/types/span.h")
  set(ABSL_INCLUDE_DIR "${LITERT_BUILD}/_deps/com_google_absl-src")
endif()
if(NOT EXISTS "${ABSL_INCLUDE_DIR}/absl/types/span.h")
  message(FATAL_ERROR "Could not find absl/types/span.h in LiteRT build tree")
endif()

# Generated LiteRT headers
if(NOT EXISTS "${LITERT_BUILD}/include/litert/build_common/build_config.h")
  message(FATAL_ERROR "Missing ${LITERT_BUILD}/include/litert/build_common/build_config.h")
endif()

add_executable(litert_cpu_bench main.cpp)

target_include_directories(litert_cpu_bench PRIVATE
  "${LITERT_ROOT}"           # for #include "litert/cc/..."
  "${LITERT_BUILD}/include"  # generated headers
  "${ABSL_INCLUDE_DIR}"      # Abseil headers
)

# Parse run_model link command and reuse its libs/paths/options safely.
file(READ "${RUN_MODEL_LINK_TXT}" RUN_MODEL_LINK_CMD)
separate_arguments(RUN_MODEL_LINK_ARGS UNIX_COMMAND "${RUN_MODEL_LINK_CMD}")
get_filename_component(RUN_MODEL_LINK_DIR "${RUN_MODEL_LINK_TXT}" DIRECTORY)

set(REUSED_LINK_DIRS "")
set(REUSED_LINK_LIBS "")
set(REUSED_LINK_OPTS "")

foreach(arg IN LISTS RUN_MODEL_LINK_ARGS)
  # Keep useful linker options
  if(arg STREQUAL "-pthread" OR arg MATCHES "^-Wl,.+")
    list(APPEND REUSED_LINK_OPTS "${arg}")
    continue()
  endif()

  # Link search dirs
  if(arg MATCHES "^-L.+")
    string(REGEX REPLACE "^-L" "" _ldir "${arg}")
    if(IS_ABSOLUTE "${_ldir}")
      list(APPEND REUSED_LINK_DIRS "${_ldir}")
    else()
      get_filename_component(_abs_ldir "${RUN_MODEL_LINK_DIR}/${_ldir}" ABSOLUTE)
      list(APPEND REUSED_LINK_DIRS "${_abs_ldir}")
    endif()
    continue()
  endif()

  # Absolute lib files
  if(arg MATCHES "^/.*\\.(a|so)(\\.[0-9]+)*$")
    list(APPEND REUSED_LINK_LIBS "${arg}")
    continue()
  endif()

  # Relative lib files
  if(arg MATCHES "^(\\./|\\.\\./).+\\.(a|so)(\\.[0-9]+)*$")
    get_filename_component(_abs_rel "${RUN_MODEL_LINK_DIR}/${arg}" ABSOLUTE)
    list(APPEND REUSED_LINK_LIBS "${_abs_rel}")
    continue()
  endif()

  # Malformed '-l../path/libxxx.a' style from link.txt -> convert to file path
  if(arg MATCHES "^-l(.+/.*\\.(a|so)(\\.[0-9]+)*)$")
    string(REGEX REPLACE "^-l" "" _libpath "${arg}")
    get_filename_component(_abs_lib "${RUN_MODEL_LINK_DIR}/${_libpath}" ABSOLUTE)
    list(APPEND REUSED_LINK_LIBS "${_abs_lib}")
    continue()
  endif()

  # Normal -lfoo
  if(arg MATCHES "^-l[^/]+$")
    list(APPEND REUSED_LINK_LIBS "${arg}")
    continue()
  endif()
endforeach()

list(REMOVE_DUPLICATES REUSED_LINK_DIRS)

if(REUSED_LINK_LIBS STREQUAL "")
  message(FATAL_ERROR "Could not extract link libs from ${RUN_MODEL_LINK_TXT}")
endif()

if(REUSED_LINK_DIRS)
  target_link_directories(litert_cpu_bench PRIVATE ${REUSED_LINK_DIRS})
endif()

if(REUSED_LINK_OPTS)
  target_link_options(litert_cpu_bench PRIVATE ${REUSED_LINK_OPTS})
endif()

target_link_libraries(litert_cpu_bench PRIVATE ${REUSED_LINK_LIBS})

set_target_properties(litert_cpu_bench PROPERTIES
  BUILD_RPATH
    "${LITERT_BUILD};${LITERT_BUILD}/c;${LITERT_BUILD}/cc;${LITERT_BUILD}/runtime;${LITERT_BUILD}/tools"
)