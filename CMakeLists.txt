cmake_minimum_required(VERSION 3.9.6)

project(tmp LANGUAGES CXX )

find_package(OpenMP REQUIRED)

include(FetchContent)

FetchContent_Declare(windowblit URL "https://github.com/tay10r/windowblit/archive/main.zip")
FetchContent_MakeAvailable(windowblit)

set(sources
  main.cpp
  common.hpp
  common.cpp)

if(CMAKE_CUDA_COMPILER)
  list(APPEND sources
    render.hpp
    render.cu)
else(CMAKE_CUDA_COMPILER)
  list(APPEND sources
    render.hpp
    render.cpp)
endif(CMAKE_CUDA_COMPILER)

add_executable(main ${sources})

if(NOT MSVC)
  target_compile_options(main PRIVATE -Wall -Wextra -Werror -Wfatal-errors)
endif(NOT MSVC)

target_compile_features(main PRIVATE cxx_std_17)

target_link_libraries(main PRIVATE window_blit OpenMP::OpenMP_CXX)
