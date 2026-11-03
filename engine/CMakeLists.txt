cmake_minimum_required(VERSION 3.25)
project(rookery CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8)
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2")
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
  # The popcount and bit-scan instructions; every x86-64 CPU from the last fifteen years has them.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    add_compile_options(-mpopcnt -mbmi)
  endif()
  # For playing on this machine: lets the compiler vectorise the network with AVX2 and the like.
  option(ROOKERY_NATIVE "Optimise for the CPU building it" OFF)
  if(ROOKERY_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

file(GLOB CORE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
list(REMOVE_ITEM CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
add_library(rookery_core STATIC ${CORE_SOURCES})
target_include_directories(rookery_core PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(rookery_core PUBLIC Threads::Threads)
if(MINGW)
  # A self-contained executable: no MinGW runtime DLLs needed next to it.
  target_link_options(rookery_core PUBLIC -static)
endif()

add_executable(rookery src/main.cpp)
target_link_libraries(rookery PRIVATE rookery_core)

enable_testing()
add_executable(rookery_tests tests/tests.cpp)
target_link_libraries(rookery_tests PRIVATE rookery_core)
target_compile_definitions(rookery_tests PRIVATE ROOKERY_NETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../nets")
add_test(NAME rookery_tests COMMAND rookery_tests)

# The network sits next to the executable, where the engine looks for it by default.
set(ROOKERY_NET ${CMAKE_CURRENT_SOURCE_DIR}/../nets/rookery.nnue)
if(EXISTS ${ROOKERY_NET})
  add_custom_command(TARGET rookery POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ROOKERY_NET} $<TARGET_FILE_DIR:rookery>/rookery.nnue)
endif()
