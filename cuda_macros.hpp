#pragma once

#ifdef __CUDACC__

#define DEV_FUNC __device

#define DEVHOST_FUNC __device__ __host__

#else // __CUDACC__

#define DEV_FUNC

#define DEVHOST_FUNC

#endif // __CUDACC__
