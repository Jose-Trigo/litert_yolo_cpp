cmake_minimum_required(VERSION 3.20)

# Force clang on fresh build dirs
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
set(LITERT_SRC  "${LITERT_ROOT}/litert")
set(LITERT_BUILD "${LITERT_SRC}/build-pi-clang")
set(RUN_MODEL_LINK_TXT "${LITERT_BUILD}/tools/CMakeFiles/run_model.dir/link.txt")

if(NOT EXISTS "${RUN_MODEL_LINK_TXT}")
  message(FATAL_ERROR
    "Missing ${RUN_MODEL_LINK_TXT}\n"
    "Build it first:\n"
    "  cd ${LITERT_SRC} && cmake --build ${LITERT_BUILD} --target run_model -j")
endif()

# Headers
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

if(NOT EXISTS "${LITERT_BUILD}/include/litert/build_common/build_config.h")
  message(FATAL_ERROR "Missing ${LITERT_BUILD}/include/litert/build_common/build_config.h")
endif()

add_executable(litert_cpu_bench main.cpp)

target_include_directories(litert_cpu_bench PRIVATE
  "${LITERT_ROOT}"
  "${LITERT_BUILD}/include"
  "${ABSL_INCLUDE_DIR}"
)

# Parse run_model link command and reuse link items.
file(READ "${RUN_MODEL_LINK_TXT}" RUN_MODEL_LINK_CMD)
separate_arguments(RUN_MODEL_LINK_ARGS UNIX_COMMAND "${RUN_MODEL_LINK_CMD}")

# IMPORTANT: run_model's link command is relative to this dir
set(RUN_MODEL_WORKDIR "${LITERT_BUILD}/tools")
get_filename_component(RUN_MODEL_LINK_DIR "${RUN_MODEL_LINK_TXT}" DIRECTORY)

set(REUSED_LINK_DIRS "")
set(REUSED_LINK_LIBS "")
set(REUSED_LINK_OPTS "")

function(resolve_existing_path out_var p)
  if(IS_ABSOLUTE "${p}" AND EXISTS "${p}")
    set(${out_var} "${p}" PARENT_SCOPE)
    return()
  endif()

  set(c1 "${RUN_MODEL_WORKDIR}/${p}")
  set(c2 "${RUN_MODEL_LINK_DIR}/${p}")
  set(c3 "${LITERT_BUILD}/${p}")

  if(EXISTS "${c1}")
    set(${out_var} "${c1}" PARENT_SCOPE)
  elseif(EXISTS "${c2}")
    set(${out_var} "${c2}" PARENT_SCOPE)
  elseif(EXISTS "${c3}")
    set(${out_var} "${c3}" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

foreach(arg IN LISTS RUN_MODEL_LINK_ARGS)
  # Useful linker options
  if(arg STREQUAL "-pthread" OR arg MATCHES "^-Wl,.+")
    list(APPEND REUSED_LINK_OPTS "${arg}")
    continue()
  endif()

  # -L search paths
  if(arg MATCHES "^-L.+")
    string(REGEX REPLACE "^-L" "" ldir "${arg}")
    resolve_existing_path(abs_ldir "${ldir}")
    if(NOT abs_ldir STREQUAL "")
      list(APPEND REUSED_LINK_DIRS "${abs_ldir}")
    endif()
    continue()
  endif()

  # Absolute lib file
  if(arg MATCHES "^/.*\\.(a|so)(\\.[0-9]+)*$")
    if(EXISTS "${arg}")
      list(APPEND REUSED_LINK_LIBS "${arg}")
    endif()
    continue()
  endif()

  # Relative lib file
  if(arg MATCHES "^(\\./|\\.\\./).+\\.(a|so)(\\.[0-9]+)*$")
    resolve_existing_path(abs_rel_lib "${arg}")
    if(NOT abs_rel_lib STREQUAL "")
      list(APPEND REUSED_LINK_LIBS "${abs_rel_lib}")
    endif()
    continue()
  endif()

  # Malformed "-l../path/libxxx.a" style
  if(arg MATCHES "^-l(.+/.*\\.(a|so)(\\.[0-9]+)*)$")
    string(REGEX REPLACE "^-l" "" libpath "${arg}")
    resolve_existing_path(abs_malformed_lib "${libpath}")
    if(NOT abs_malformed_lib STREQUAL "")
      list(APPEND REUSED_LINK_LIBS "${abs_malformed_lib}")
    else()
      message(FATAL_ERROR "Could not resolve library path from: ${arg}")
    endif()
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

# Critical for static archives with circular deps
target_link_libraries(litert_cpu_bench PRIVATE
  -Wl,--start-group
  ${REUSED_LINK_LIBS}
  -Wl,--end-group
)

set_target_properties(litert_cpu_bench PROPERTIES
  BUILD_RPATH
    "${LITERT_BUILD};${LITERT_BUILD}/c;${LITERT_BUILD}/cc;${LITERT_BUILD}/runtime;${LITERT_BUILD}/tools"
)