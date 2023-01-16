cmake_minimum_required(VERSION 3.10)

project(cuda-crop CUDA CXX)

add_definitions(-DAPI_EXPORTS)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 87)
enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("-- Running on an embedded Tegra platform")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("-- Running on standard Linux (non Tegra)")
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)
endif()

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(centre-crop ${PROJECT_SOURCE_DIR}/src/centre_crop.cpp ${SRCS})
target_compile_options(centre-crop
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17 -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++11 -gencode arch=compute_87,code=sm_87 -Xptxas -O3>
)

target_link_libraries(centre-crop ${OpenCV_LIBS})
