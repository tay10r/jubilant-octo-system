cmake_minimum_required(VERSION 3.9.6)

project(tmp LANGUAGES CXX)

option(ENABLE_CUDA "Whether or not to enable CUDA." ON)

if(ENABLE_CUDA)
  enable_language(CUDA)
endif(ENABLE_CUDA)

find_package(OpenMP REQUIRED)

include(FetchContent)

FetchContent_Declare(windowblit URL "https://github.com/tay10r/windowblit/archive/main.zip")
FetchContent_MakeAvailable(windowblit)

set(sources
  main.cpp
  src/bvh.cpp)

if(CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
  list(APPEND sources src/render.cu src/bvh.cu)
else(CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
  list(APPEND sources src/render.cpp)
endif(CMAKE_CUDA_COMPILER AND ENABLE_CUDA)

add_executable(main ${sources})

target_compile_features(main PRIVATE cxx_std_17)

target_link_libraries(main PRIVATE window_blit OpenMP::OpenMP_CXX)

target_include_directories(main PRIVATE include)
