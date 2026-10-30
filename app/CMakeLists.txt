cmake_minimum_required(VERSION 3.16)
project(dronolok_app CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- ядро детектора: собираем вместе с приложением ---
set(DRONOLOK_TESTS OFF CACHE BOOL "Тесты ядер")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../native ${CMAKE_BINARY_DIR}/native)

# --- Dear ImGui: локальная копия (IMGUI_DIR) или скачивание ---
set(IMGUI_DIR "" CACHE PATH "Папка с исходниками Dear ImGui (пусто = скачать)")
if(NOT IMGUI_DIR)
  include(FetchContent)
  FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9b
    GIT_SHALLOW TRUE)
  FetchContent_GetProperties(imgui)
  if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
  endif()
  set(IMGUI_DIR ${imgui_SOURCE_DIR})
endif()

# --- SDL2 ---
find_package(SDL2 REQUIRED)
if(TARGET SDL2::SDL2-static AND (MINGW OR DRONOLOK_STATIC_SDL))
  set(SDL_LIBS SDL2::SDL2-static)
else()
  set(SDL_LIBS SDL2::SDL2)
endif()
if(TARGET SDL2::SDL2main)
  list(PREPEND SDL_LIBS SDL2::SDL2main)
endif()

add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
  ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer2.cpp)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC ${SDL_LIBS})

# --- FFmpeg (libavformat/libavcodec/libswscale/libavutil) через pkg-config ---
find_package(PkgConfig REQUIRED)
if(MINGW)
  # статическая сборка: тянем и зависимости FFmpeg (bcrypt, ws2_32 и т.п.)
  set(PKG_CONFIG_EXECUTABLE ${PKG_CONFIG_EXECUTABLE} --static)
endif()
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libswscale libavutil)

add_executable(dronolok_app
  src/annotate.cpp
  src/main.cpp
  src/multitest.cpp
  src/multitest_report.cpp
  src/pipeline.cpp
  src/platform.cpp
  src/scenelink.cpp
  src/twocam.cpp
  src/video_source.cpp)
target_include_directories(dronolok_app PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/../native/tools ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(dronolok_app PRIVATE dronolok_static imgui)
if(MINGW)
  target_link_libraries(dronolok_app PRIVATE ${FFMPEG_STATIC_LDFLAGS} comdlg32 ole32 shell32 psapi advapi32 winmm ws2_32)
  target_link_options(dronolok_app PRIVATE -static -mwindows)
  set_target_properties(dronolok_app PROPERTIES OUTPUT_NAME "Dronolok")
else()
  target_link_libraries(dronolok_app PRIVATE ${FFMPEG_LDFLAGS})
endif()
if(MSVC)
  target_compile_options(dronolok_app PRIVATE /O2 /W3 /utf-8)
else()
  target_compile_options(dronolok_app PRIVATE -O2 -Wall -Wextra -Wno-missing-field-initializers)
endif()
