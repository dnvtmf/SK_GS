/* adapt from nvdiffrec/render/renderutils/c_src
 *
 * Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once
// #include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include "util.cuh"

#define BLOCK_X 8
#define BLOCK_Y 8

#define NVDR_CHECK_GL_ERROR(GL_CALL)                                                               \
  {                                                                                                \
    GL_CALL;                                                                                       \
    GLenum err = glGetError();                                                                     \
    TORCH_CHECK(err == GL_NO_ERROR, "OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]"); \
  }
#define CHECK_TENSOR(X, DIMS, CHANNELS)                                                                            \
  TORCH_CHECK(X.is_cuda(), #X " must be a cuda tensor")                                                            \
  TORCH_CHECK(X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kBFloat16, #X " must be fp32 or bf16") \
  TORCH_CHECK(X.dim() == DIMS, #X " must have " #DIMS " dimensions")                                               \
  TORCH_CHECK(X.size(DIMS - 1) == CHANNELS, #X " must have " #CHANNELS " channels")

struct vec3f {
  float x, y, z;

#ifdef __CUDACC__
  __device__ vec3f() {}
  __device__ vec3f(float v) {
    x = v;
    y = v;
    z = v;
  }
  __device__ vec3f(float _x, float _y, float _z) {
    x = _x;
    y = _y;
    z = _z;
  }
  __device__ vec3f(float3 v) {
    x = v.x;
    y = v.y;
    z = v.z;
  }

  __device__ inline vec3f& operator+=(const vec3f& b) {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
  __device__ inline vec3f& operator-=(const vec3f& b) {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }
  __device__ inline vec3f& operator*=(const vec3f& b) {
    x *= b.x;
    y *= b.y;
    z *= b.z;
    return *this;
  }
  __device__ inline vec3f& operator/=(const vec3f& b) {
    x /= b.x;
    y /= b.y;
    z /= b.z;
    return *this;
  }
#endif
};

#ifdef __CUDACC__
__device__ static inline vec3f operator+(const vec3f& a, const vec3f& b) {
  return vec3f(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ static inline vec3f operator-(const vec3f& a, const vec3f& b) {
  return vec3f(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ static inline vec3f operator*(const vec3f& a, const vec3f& b) {
  return vec3f(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ static inline vec3f operator/(const vec3f& a, const vec3f& b) {
  return vec3f(a.x / b.x, a.y / b.y, a.z / b.z);
}
__device__ static inline vec3f operator-(const vec3f& a) { return vec3f(-a.x, -a.y, -a.z); }

__device__ static inline float sum(vec3f a) { return a.x + a.y + a.z; }

__device__ static inline vec3f cross(vec3f a, vec3f b) {
  vec3f out;
  out.x = a.y * b.z - a.z * b.y;
  out.y = a.z * b.x - a.x * b.z;
  out.z = a.x * b.y - a.y * b.x;
  return out;
}

__device__ static inline void bwdCross(vec3f a, vec3f b, vec3f& d_a, vec3f& d_b, vec3f d_out) {
  d_a.x += d_out.z * b.y - d_out.y * b.z;
  d_a.y += d_out.x * b.z - d_out.z * b.x;
  d_a.z += d_out.y * b.x - d_out.x * b.y;

  d_b.x += d_out.y * a.z - d_out.z * a.y;
  d_b.y += d_out.z * a.x - d_out.x * a.z;
  d_b.z += d_out.x * a.y - d_out.y * a.x;
}

__device__ static inline float dot(vec3f a, vec3f b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ static inline void bwdDot(vec3f a, vec3f b, vec3f& d_a, vec3f& d_b, float d_out) {
  d_a.x += d_out * b.x;
  d_a.y += d_out * b.y;
  d_a.z += d_out * b.z;
  d_b.x += d_out * a.x;
  d_b.y += d_out * a.y;
  d_b.z += d_out * a.z;
}

__device__ static inline vec3f reflect(vec3f x, vec3f n) { return n * 2.0f * dot(n, x) - x; }

__device__ static inline void bwdReflect(vec3f x, vec3f n, vec3f& d_x, vec3f& d_n, const vec3f d_out) {
  d_x.x += d_out.x * (2 * n.x * n.x - 1) + d_out.y * (2 * n.x * n.y) + d_out.z * (2 * n.x * n.z);
  d_x.y += d_out.x * (2 * n.x * n.y) + d_out.y * (2 * n.y * n.y - 1) + d_out.z * (2 * n.y * n.z);
  d_x.z += d_out.x * (2 * n.x * n.z) + d_out.y * (2 * n.y * n.z) + d_out.z * (2 * n.z * n.z - 1);

  d_n.x +=
      d_out.x * (2 * (2 * n.x * x.x + n.y * x.y + n.z * x.z)) + d_out.y * (2 * n.y * x.x) + d_out.z * (2 * n.z * x.x);
  d_n.y +=
      d_out.x * (2 * n.x * x.y) + d_out.y * (2 * (n.x * x.x + 2 * n.y * x.y + n.z * x.z)) + d_out.z * (2 * n.z * x.y);
  d_n.z +=
      d_out.x * (2 * n.x * x.z) + d_out.y * (2 * n.y * x.z) + d_out.z * (2 * (n.x * x.x + n.y * x.y + 2 * n.z * x.z));
}

__device__ static inline vec3f safeNormalize(vec3f v) {
  float l = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  return l > 0.0f ? (v / l) : vec3f(0.0f);
}

__device__ static inline void bwdSafeNormalize(const vec3f v, vec3f& d_v, const vec3f d_out) {
  float l = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  if (l > 0.0f) {
    float fac = 1.0 / powf(v.x * v.x + v.y * v.y + v.z * v.z, 1.5f);
    d_v.x += (d_out.x * (v.y * v.y + v.z * v.z) - d_out.y * (v.x * v.y) - d_out.z * (v.x * v.z)) * fac;
    d_v.y += (d_out.y * (v.x * v.x + v.z * v.z) - d_out.x * (v.y * v.x) - d_out.z * (v.y * v.z)) * fac;
    d_v.z += (d_out.z * (v.x * v.x + v.y * v.y) - d_out.x * (v.z * v.x) - d_out.y * (v.z * v.y)) * fac;
  }
}

#endif

struct vec4f {
  float x, y, z, w;

#ifdef __CUDACC__
  __device__ vec4f() {}
  __device__ vec4f(float v) {
    x = v;
    y = v;
    z = v;
    w = v;
  }
  __device__ vec4f(float _x, float _y, float _z, float _w) {
    x = _x;
    y = _y;
    z = _z;
    w = _w;
  }
  __device__ vec4f(float4 v) {
    x = v.x;
    y = v.y;
    z = v.z;
    w = v.w;
  }
#endif
};

//---------------------------------------------------------------------------------
// CUDA-side Tensor class for in/out parameter parsing. Can be float32 or bfloat16

struct RenderTensor {
  void* val;
  void* d_val;
  int dims[4], _dims[4];
  int strides[4];
  bool fp16;

#if defined(__CUDA__) && !defined(__CUDA_ARCH__)
  RenderTensor() : val(nullptr), d_val(nullptr), fp16(true), dims{0, 0, 0, 0}, _dims{0, 0, 0, 0}, strides{0, 0, 0, 0} {}
#endif

#ifdef __CUDACC__
  // Helpers to index and read/write a single element
  __device__ inline int _nhwcIndex(int n, int h, int w, int c) const {
    return n * strides[0] + h * strides[1] + w * strides[2] + c * strides[3];
  }
  __device__ inline int nhwcIndex(int n, int h, int w, int c) const {
    return (dims[0] == 1 ? 0 : n * strides[0]) + (dims[1] == 1 ? 0 : h * strides[1]) +
           (dims[2] == 1 ? 0 : w * strides[2]) + (dims[3] == 1 ? 0 : c * strides[3]);
  }
  __device__ inline int nhwcIndexContinuous(int n, int h, int w, int c) const {
    return ((n * _dims[1] + h) * _dims[2] + w) * _dims[3] + c;
  }
#ifdef BFLOAT16
  __device__ inline float fetch(unsigned int idx) const {
    return fp16 ? __bfloat162float(((__nv_bfloat16*) val)[idx]) : ((float*) val)[idx];
  }
  __device__ inline void store(unsigned int idx, float _val) {
    if (fp16)
      ((__nv_bfloat16*) val)[idx] = __float2bfloat16(_val);
    else
      ((float*) val)[idx] = _val;
  }
  __device__ inline void store_grad(unsigned int idx, float _val) {
    if (fp16)
      ((__nv_bfloat16*) d_val)[idx] = __float2bfloat16(_val);
    else
      ((float*) d_val)[idx] = _val;
  }
#else
  __device__ inline float fetch(unsigned int idx) const { return ((float*) val)[idx]; }
  __device__ inline void store(unsigned int idx, float _val) { ((float*) val)[idx] = _val; }
  __device__ inline void store_grad(unsigned int idx, float _val) { ((float*) d_val)[idx] = _val; }
#endif

  //////////////////////////////////////////////////////////////////////////////////////////
  // Fetch, use broadcasting for tensor dimensions of size 1
  __device__ inline float fetch1(unsigned int x, unsigned int y, unsigned int z) const {
    return fetch(nhwcIndex(z, y, x, 0));
  }

  __device__ inline vec3f fetch3(unsigned int x, unsigned int y, unsigned int z) const {
    return vec3f(fetch(nhwcIndex(z, y, x, 0)), fetch(nhwcIndex(z, y, x, 1)), fetch(nhwcIndex(z, y, x, 2)));
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Store, no broadcasting here. Assume we output full res gradient and then reduce using torch.sum outside
  __device__ inline void store(unsigned int x, unsigned int y, unsigned int z, float _val) {
    store(_nhwcIndex(z, y, x, 0), _val);
  }

  __device__ inline void store(unsigned int x, unsigned int y, unsigned int z, vec3f _val) {
    store(_nhwcIndex(z, y, x, 0), _val.x);
    store(_nhwcIndex(z, y, x, 1), _val.y);
    store(_nhwcIndex(z, y, x, 2), _val.z);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Store gradient , no broadcasting here. Assume we output full res gradient and then reduce using torch.sum outside
  __device__ inline void store_grad(unsigned int x, unsigned int y, unsigned int z, float _val) {
    store_grad(nhwcIndexContinuous(z, y, x, 0), _val);
  }

  __device__ inline void store_grad(unsigned int x, unsigned int y, unsigned int z, vec3f _val) {
    store_grad(nhwcIndexContinuous(z, y, x, 0), _val.x);
    store_grad(nhwcIndexContinuous(z, y, x, 1), _val.y);
    store_grad(nhwcIndexContinuous(z, y, x, 2), _val.z);
  }
#endif
};

void update_grid(dim3& gridSize, torch::Tensor x);
template <typename... Ts>
void update_grid(dim3& gridSize, torch::Tensor x, Ts&&... vs) {
  gridSize.x = std::max(gridSize.x, (uint32_t) x.size(2));
  gridSize.y = std::max(gridSize.y, (uint32_t) x.size(1));
  gridSize.z = std::max(gridSize.z, (uint32_t) x.size(0));
  update_grid(gridSize, std::forward<Ts>(vs)...);
}
RenderTensor make_cuda_tensor(torch::Tensor val);
RenderTensor make_cuda_tensor(torch::Tensor val, dim3 outDims, torch::Tensor* grad = nullptr);

dim3 getLaunchBlockSize(int maxWidth, int maxHeight, dim3 dims);
dim3 getLaunchGridSize(dim3 blockSize, dim3 dims);

#ifdef __CUDACC__

#ifdef _MSC_VER
#define M_PI 3.14159265358979323846f
#endif

__host__ __device__ static inline dim3 getWarpSize(dim3 blockSize) {
  return dim3(min(blockSize.x, 32u), min(max(32u / blockSize.x, 1u), min(32u, blockSize.y)),
      min(max(32u / (blockSize.x * blockSize.y), 1u), min(32u, blockSize.z)));
}

__device__ static inline float clamp(float val, float mn, float mx) { return min(max(val, mn), mx); }
#else
dim3 getWarpSize(dim3 blockSize);
#endif
