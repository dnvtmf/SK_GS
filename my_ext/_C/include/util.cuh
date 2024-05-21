#pragma once
// #include <cooperative_groups.h>
#if defined(__INTELLISENSE__) && !defined(__CUDACC__)
#define __CUDACC__
#endif

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__) && defined(BFLOAT16)
#include <cuda_bf16.h>  // bfloat16 is float32 compatible with less mantissa bits
#endif

#include "common.hpp"
#include "sort.cuh"

// for vscode
#ifdef __INTELLISENSE__
#define KERNEL_ARG(...)
#else
#define KERNEL_ARG(...) <<< __VA_ARGS__ >>>
#endif

#if QLEN == 32
#define POPCNT __popc
#else
#define POPCNT __popcll
#endif

#define CHECK_CUDA_ERROR(s)                                               \
  {                                                                       \
    auto error = cudaGetLastError();                                      \
    BCNN_ASSERT(error == cudaSuccess, s, " ", cudaGetErrorString(error)); \
  }

#define CUDA_NUM_THREADS 1024
#define WARP_SIZE 32

#define GET_BIT(x, pos) (((x) >> (pos)) & 1)

inline int get_cuda_threads(int n) {
  for (int i = 32; i != CUDA_NUM_THREADS; i *= 2) {
    if (i >= n) return i;
  }
  return CUDA_NUM_THREADS;
}

template <typename T>
__device__ __forceinline__ void _swap(T &a, T &b) {
  T t = a;
  a   = b;
  b   = t;
}

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

template <typename T>
__device__ __forceinline__ T clamp(const T &x, const T &min_value = 0., const T &max_value = 1.) {
  return min(max(x, min_value), max_value);
}

// len <= 4, blockDim.x >= 32
template <typename T>
__device__ void small_matrix_inv(T *m, T *r, int len) {
  int tid = threadIdx.x;
  T t;
  int x = tid / len, y = tid % len;
  int i, j, l;
  if (tid < len * len) r[tid] = x == y;  // init
  for (l = 0; l < len; ++l) {
    // arg max
    if (tid == 0) {
      for (i = l + 1, j = l; i < len; ++i) {
        if (m[i * len + l] > m[j * len + l]) j = i;
      }
    }
    j = __shfl_sync(0xffffffff, j, 0);
    // swap row j and row l
    if (j != l && tid < len) {
      t                = m[l * len + tid];
      m[l * len + tid] = m[j * len + tid];
      m[j * len + tid] = t;
      t                = r[l * len + tid];
      r[l * len + tid] = r[j * len + tid];
      r[j * len + tid] = t;
    }
    // normalize l-th row
    if (tid < len) {
      t = T(1.) / m[l * len + l];
      m[l * len + tid] *= t;
      r[l * len + tid] *= t;
    }
    // cacluate
    if (x < len && x != l && m[x * len + l] != 0) {
      t = m[x * len + l];
      m[x * len + y] -= m[l * len + y] * t;
      r[x * len + y] -= r[l * len + y] * t;
    }
  }
}

// 4 < len <= 32, blockDimx.x >= len * 32
template <typename T>
__device__ void middle_matrix_inv(T *m, T *r, int len) {
  int tid = threadIdx.x;
  T t;
  int x = tid / WARP_SIZE, y = tid % WARP_SIZE;
  int i, j, l;
  volatile __shared__ int row;
  if (y < len) r[x * len + y] = x == y;  // init
  for (l = 0; l < len; ++l) {
    // arg max
    if (tid == 0) {
      for (i = l + 1, j = l; i < len; ++i) {
        if (m[i * len + l] > m[j * len + l]) j = i;
      }
      row = j;
    }
    __syncthreads();
    if (tid < len) {
      // swap row j and row l
      j = row;
      if (j != l) {
        t              = m[l * len + y];
        m[l * len + y] = m[j * len + y];
        m[j * len + y] = t;
        t              = r[l * len + y];
        r[l * len + y] = r[j * len + y];
        r[j * len + y] = t;
      }
      // normalize l-th row
      t = T(1.) / m[l * len + l];
      m[l * len + y] *= t;
      r[l * len + y] *= t;
    }
    __syncthreads();
    // cacluate
    if (x < len && x != l && y < len && m[x * len + l] != 0) {
      t = m[x * len + l];
      m[x * len + y] -= m[l * len + y] * t;
      r[x * len + y] -= r[l * len + y] * t;
    }
    __syncthreads();
  }
}

template <typename T>
__device__ __forceinline__ T shfl_xor(T v, int i) {
#if __CUDA_ARCH__ >= 300
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, v, i, WARP_SIZE);
#else
  return __shfl_xor(v, i, WARP_SIZE);
#endif
#else
  assert(false);
#endif
}

template <typename T, bool broadcast = true>
__device__ __forceinline__ void reduce_sum_wrap(T &val) {
#pragma unroll
  for (int i = 16; i >= 1; i /= 2) val += shfl_xor<T>(val, i);
  if (broadcast) val = __shfl_sync(0xffffffff, val, 0);
}

// If you use two or more reudce_sum_block(), please be carefully! Add __syncthreads() between them.
template <typename T, bool broadcast = true>
__device__ void reduce_sum_block(T &val) {
  int tid     = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  volatile __shared__ T smem[WARP_SIZE];
#pragma unroll
  for (int i = 16; i >= 1; i /= 2) val += shfl_xor<T>(val, i);
  if (lane_id == 0) smem[tid / WARP_SIZE] = val;
  if (tid < WARP_SIZE && tid >= (blockDim.x / WARP_SIZE)) smem[tid] = 0;
  __syncthreads();
  if (tid < WARP_SIZE) {
    val = smem[tid];
#pragma unroll
    for (int i = 16; i >= 1; i /= 2) val += shfl_xor<T>(val, i);
  }
  if (broadcast) {
    if (tid == 0) smem[0] = val;
    __syncthreads();
    val = smem[0];
  }
}

template <typename T, bool broadcast = true>
__device__ void reduce_max_block(T &val) {
  int tid     = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  volatile __shared__ T smem[WARP_SIZE];
#pragma unroll
  for (int i = WARP_SIZE / 2; i >= 1; i >>= 1) val = max(val, shfl_xor<T>(val, i));
  if (lane_id == 0) smem[tid / WARP_SIZE] = val;
  if (tid < WARP_SIZE && tid >= blockDim.x / WARP_SIZE) smem[tid] = val;  // avoid undefined data in smem
  __syncthreads();
  if (tid < WARP_SIZE) {
    val = smem[tid];
#pragma unroll
    for (int i = WARP_SIZE / 2; i >= 1; i >>= 1) val = max(val, shfl_xor<T>(val, i));
  }
  if (broadcast) {
    if (tid == 0) smem[0] = val;
    __syncthreads();
    val = smem[0];
  }
}

template <typename T, bool broadcast = true>
__device__ void reduce_min_block(T &val) {
  int tid     = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  volatile __shared__ T smem[WARP_SIZE];
#pragma unroll
  for (int i = WARP_SIZE / 2; i >= 1; i >>= 1) val = min(val, shfl_xor<T>(val, i));
  if (lane_id == 0) smem[tid / WARP_SIZE] = val;
  if (tid < WARP_SIZE && tid >= blockDim.x / WARP_SIZE) smem[tid] = val;  // avoid undefined data in smem
  __syncthreads();
  if (tid < WARP_SIZE) {
    val = smem[tid];
#pragma unroll
    for (int i = WARP_SIZE / 2; i >= 1; i >>= 1) val = min(val, shfl_xor<T>(val, i));
  }
  if (broadcast) {
    if (tid == 0) smem[0] = val;
    __syncthreads();
    val = smem[0];
  }
}

template <typename T>
__device__ __forceinline__ T reduce2(T val) {
  int tid = threadIdx.x;
  volatile __shared__ T smem[CUDA_NUM_THREADS - WARP_SIZE];
  if (tid >= WARP_SIZE) smem[tid - WARP_SIZE] = val;
  __syncthreads();
  if (tid < WARP_SIZE) {
    for (int k = tid; k < CUDA_NUM_THREADS - WARP_SIZE; k += WARP_SIZE) {
      val += smem[k];
    }
    smem[tid] = val;
  }
  if (tid < 16) smem[tid] += smem[tid + 16];
  if (tid < 8) smem[tid] += smem[tid + 8];
  if (tid < 4) smem[tid] += smem[tid + 4];
  if (tid < 2) smem[tid] += smem[tid + 2];
  if (tid < 1) smem[tid] += smem[tid + 1];
  __syncthreads();
  return smem[0];
}

template <typename T>
__device__ T scan_block(T value) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  __shared__ T sums[WARP_SIZE];
#pragma unroll
  for (int i = 1; i <= warpSize; i *= 2) {
    T t = __shfl_up_sync(0xffffffff, value, i);
    if (lane_id >= i) value += t;
  }
  if (lane_id == warpSize - 1) sums[warp_id] = value;
  if (warp_id == 0 && lane_id >= (blockDim.x / warpSize)) sums[lane_id] = 0;
  __syncthreads();
  if (warp_id == 0) {
    T warp_sum = sums[lane_id];
    for (int i = 1; i <= warpSize; i *= 2) {
      T t = __shfl_up_sync(0xffffffff, warp_sum, i);
      if (lane_id >= i) warp_sum += t;
    }
    sums[lane_id] = warp_sum;
  }
  __syncthreads();
  T blockSum = 0;
  if (warp_id > 0) blockSum = sums[warp_id - 1];
  value += blockSum;
  return value;
}

template <typename T>
__device__ __forceinline__ T scan_warp(T value) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
#pragma unroll
  for (int i = 1; i <= warpSize; i *= 2) {
    T t = __shfl_up_sync(0xffffffff, value, i);
    if (lane_id >= i) value += t;
  }
  return value;
}
