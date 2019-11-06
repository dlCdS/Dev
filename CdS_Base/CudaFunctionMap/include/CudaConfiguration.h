#pragma once
/* Configuration file:
Contains global includes
Contains defines for pre-processor options
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// __USE_CUDA__
// Flag for cuda usage

#define __USE_CUDA__

#ifdef __USE_CUDA__

#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_ERROR cudaError_t
#define CUDA_SUCCESS cudaSuccess

#define THREADS_PER_BLOCK          256

#if __CUDA_ARCH__ >= 200
#define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
#define MY_KERNEL_MIN_BLOCKS   3
#else
#define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
#define MY_KERNEL_MIN_BLOCKS   2
#endif

#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_ERROR int
#define CUDA_SUCCESS 0
#endif

#define uint unsigned int
