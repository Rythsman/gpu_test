cmake_minimum_required(VERSION 3.20)

project(micoro-benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_BUILD_TYPE Release)

set(CUDA_GEN_CODE "-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_90,code=compute_90")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

add_executable(dram_bandwitch dram_bandwidth.cu)
add_executable(dram_latency dram_latency.cu)

add_executable(l1cache_latency l1cache_latency.cu)

add_executable(l2cache_bandwidth l2cache_bandwidth.cu)
add_executable(l2cache_latency l2cache_latency.cu)

add_executable(smem_bandwidth smem_bandwidth.cu)
add_executable(smem_latency smem_latency.cu)