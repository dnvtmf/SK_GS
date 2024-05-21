#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__device__ __forceinline__ void Comparator(T &a, T &b, uint dir) {
  if ((a > b) == dir) {
    T t = a;
    a   = b;
    b   = t;
  }
}
template <typename T1, typename T2>
__device__ __forceinline__ void Comparator(T1 &keyA, T2 &valA, T1 &keyB, T2 &valB, bool dir) {
  if ((keyA > keyB) == dir) {
    T1 t1 = keyA;
    keyA  = keyB;
    keyB  = t1;
    T2 t2 = valA;
    valA  = valB;
    valB  = t2;
  }
}
/**
 * arrayLength must be 2^k <= 32
 * sortDir=0, descending, i.e. from largest to smallest
 */
template <typename T, uint sortDir = 1>
__device__ void sort_warp(T *key, uint arrayLength) {
  auto &tid = threadIdx.x;
  if (tid < arrayLength / 2) {
    for (uint size = 2; size < arrayLength; size <<= 1) {
      // Bitonic merge
      uint dir = (tid & (size / 2)) != 0;
      for (uint stride = size / 2; stride > 0; stride >>= 1) {
        // __syncthreads();
        uint pos = 2 * tid - (tid & (stride - 1));
        Comparator(key[pos], key[pos + stride], dir);
      }
    }
    // the last bitonic merge step
    for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
      // __syncthreads();
      uint pos = 2 * tid - (tid & (stride - 1));
      Comparator(key[pos], key[pos + stride], sortDir);
    }
  }
  // __syncthreads();
}
/**
 * arrayLength must be 2^k <= 1024
 * sortDir=0, descending, i.e. from largest to smallest
 */
template <typename T, uint sortDir = 1>
__device__ void sort_small(T *key, uint arrayLength) {
  auto &tid = threadIdx.x;
  for (uint size = 2; size < arrayLength; size <<= 1) {
    // Bitonic merge
    uint dir = (tid & (size / 2)) != 0;
    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      __syncthreads();
      uint pos = 2 * tid - (tid & (stride - 1));
      if (pos + stride < arrayLength) Comparator(key[pos], key[pos + stride], dir);
    }
  }
  // the last bitonic merge step
  for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    uint pos = 2 * tid - (tid & (stride - 1));
    if (pos + stride < arrayLength) Comparator(key[pos + 0], key[pos + stride], sortDir);
  }
  __syncthreads();
}
template <typename T1, typename T2, bool sortDir = 1>
__device__ void sort_small(T1 *key, T2 *val, uint arrayLength) {
  auto &tid = threadIdx.x;
  for (uint size = 2; size < arrayLength; size <<= 1) {
    // Bitonic merge
    uint dir = (tid & (size / 2)) != 0;
    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      __syncthreads();
      uint pos = 2 * tid - (tid & (stride - 1));
      if (pos + stride < arrayLength) Comparator(key[pos], val[pos], key[pos + stride], val[pos + stride], dir);
    }
  }
  // the last bitonic merge step
  for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    uint pos = 2 * tid - (tid & (stride - 1));
    if (pos + stride < arrayLength) Comparator(key[pos], val[pos], key[pos + stride], val[pos + stride], sortDir);
  }
  __syncthreads();
}
/**
 * arrayLength must be 2^k > 1024
 * sortDir=0, descending, i.e. from largest to smallest
 * It's Error when 2^k >= 8192
 */

template <typename T, uint sortDir = 1>
__device__ void sort_large(T *key, uint arrayLength) {
  auto &tid = threadIdx.x;
  for (uint size = 2; size < arrayLength; size <<= 1) {
    // Bitonic merge
    for (uint id = tid; id < arrayLength / 2; id += blockDim.x) {
      uint dir = (id & (size / 2)) != 0;
      for (uint stride = size / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        uint pos = 2 * id - (id & (stride - 1));
        if (pos + stride < arrayLength) Comparator(key[pos], key[pos + stride], dir);
      }
    }
  }
  // the last bitonic merge step
  for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    for (uint id = tid; id < arrayLength / 2; id += blockDim.x) {
      uint pos = 2 * id - (id & (stride - 1));
      if (pos + stride < arrayLength) Comparator(key[pos + 0], key[pos + stride], sortDir);
    }
  }
  __syncthreads();
}

template <typename T1, typename T2, bool sortDir = 1>
__device__ void sort_warp(T1 *s_key, T2 *s_val, uint arrayLength) {
  if (threadIdx.x < arrayLength / 2) {
    for (uint size = 2; size < arrayLength; size <<= 1) {
      // Bitonic merge
      uint dir = (threadIdx.x & (size / 2)) != 0;

      for (uint stride = size / 2; stride > 0; stride >>= 1) {
        // __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], dir);
      }
    }

    // ddd == sortDir for the last bitonic merge step
    {
      for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
        // __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], sortDir);
      }
    }
  }
  // __syncthreads();
}
