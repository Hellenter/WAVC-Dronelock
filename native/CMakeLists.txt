cmake_minimum_required(VERSION 3.16)
project(dronolok CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(DRONOLOK_TOOLS "CLI для замеров (dronolok_bench)" ON)
option(DRONOLOK_TESTS "Тесты ядер" ON)
option(DRONOLOK_SHARED "Динамическая библиотека для Python/ctypes" ON)

# --- платформа ---
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" DL_CPU)
if(DL_CPU MATCHES "^(x86_64|amd64|x64|i[3-6]86|x86)$")
  set(DL_X86 ON)
elseif(DL_CPU MATCHES "^(aarch64|arm64)")
  set(DL_ARM64 ON)
elseif(DL_CPU MATCHES "^arm")
  set(DL_ARM32 ON)
endif()

set(DL_SOURCES
  src/backend_cpu.cpp
  src/c_api.cpp
  src/cpu.cpp
  src/detector.cpp
  src/kernels_scalar.cpp
  src/kernels_sse2.cpp
  src/kernels_avx2.cpp
  src/kernels_neon.cpp
  src/motion.cpp
  src/scene.cpp
  src/thread_pool.cpp
  src/tracker.cpp
  src/silhouette.cpp
  src/tuning.cpp)

add_library(dronolok_obj OBJECT ${DL_SOURCES})
target_include_directories(dronolok_obj PUBLIC include PRIVATE src)
set_target_properties(dronolok_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(dronolok_obj PUBLIC Threads::Threads)

if(MSVC)
  target_compile_options(dronolok_obj PRIVATE /O2 /W3)
else()
  target_compile_options(dronolok_obj PRIVATE -O3 -Wall -Wextra -Wno-unused-function)
  # эталонный скалярный путь: без автовекторизации, чтобы честно сравнивать с SIMD
  set_source_files_properties(src/kernels_scalar.cpp PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize")
endif()

if(DL_X86)
  target_compile_definitions(dronolok_obj PRIVATE DL_HAVE_X86)
  if(MSVC)
    set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
  endif()
elseif(DL_ARM64)
  target_compile_definitions(dronolok_obj PRIVATE DL_HAVE_NEON)
elseif(DL_ARM32)
  # Raspberry Pi 3 с 32-битной ОС: NEON есть у Cortex-A53, но его надо включить явно
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mfpu=neon-fp-armv8" DL_HAS_MFPU)
  if(DL_HAS_MFPU)
    set_source_files_properties(src/kernels_neon.cpp PROPERTIES COMPILE_OPTIONS "-mfpu=neon-fp-armv8;-mfloat-abi=hard")
    target_compile_definitions(dronolok_obj PRIVATE DL_HAVE_NEON)
  endif()
endif()

add_library(dronolok_static STATIC $<TARGET_OBJECTS:dronolok_obj>)
target_include_directories(dronolok_static PUBLIC include)
target_link_libraries(dronolok_static PUBLIC Threads::Threads)
if(MINGW)
  target_link_options(dronolok_static INTERFACE -static)
endif()

if(DRONOLOK_SHARED)
  add_library(dronolok SHARED $<TARGET_OBJECTS:dronolok_obj>)
  target_include_directories(dronolok PUBLIC include)
  target_link_libraries(dronolok PUBLIC Threads::Threads)
  target_compile_definitions(dronolok PRIVATE DRONOLOK_SHARED)
  if(MINGW)
    target_link_options(dronolok PRIVATE -static)
    set_target_properties(dronolok PROPERTIES PREFIX "")
  endif()
endif()
if(WIN32)
  target_compile_definitions(dronolok_obj PRIVATE DRONOLOK_SHARED)
endif()

if(DRONOLOK_TOOLS)
  add_executable(dronolok_bench tools/dronolok_bench.cpp)
  target_link_libraries(dronolok_bench PRIVATE dronolok_static)
  target_include_directories(dronolok_bench PRIVATE src tools)
endif()

if(DRONOLOK_TESTS)
  enable_testing()
  add_executable(test_kernels tests/test_kernels.cpp)
  target_link_libraries(test_kernels PRIVATE dronolok_static)
  target_include_directories(test_kernels PRIVATE src tools)
  add_test(NAME kernels COMMAND test_kernels)
endif()
