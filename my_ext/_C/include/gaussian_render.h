/*
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code:  https://github.com/graphdeco-inria/diff-gaussian-rasterization
*/
/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <functional>
#include <glm/glm.hpp>
#include <vector>

#include "device_launch_parameters.h"
// #include "ops_3d.h"
namespace GaussianRasterizer {
#define NUM_CHANNELS 3  // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)

// Spherical harmonics coefficients
__device__ const float SH_C0   = 0.28209479177387814f;
__device__ const float SH_C1   = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f};
__device__ const float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
    -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid) {
  rect_min = {min(grid.x, max((int) 0, (int) ((p.x - max_radius) / BLOCK_X))),
      min(grid.y, max((int) 0, (int) ((p.y - max_radius) / BLOCK_Y)))};
  rect_max = {min(grid.x, max((int) 0, (int) ((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
      min(grid.y, max((int) 0, (int) ((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))};
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv) {
  float sum2     = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
  float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
  return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv) {
  float sum2     = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float3 dnormvdv;
  dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
  dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
  dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv) {
  float sum2     = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float4 vdv    = {v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w};
  float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
  float4 dnormvdv;
  dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
  dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
  dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
  dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ glm::vec3 computeColorFromSH(
    int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped);

__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos,
    const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs);

class Rasterizer {
 public:
  static void markVisible(int P, float* means3D, float* viewmatrix, float* projmatrix, bool* present);

  static int forward(std::function<char*(size_t)> geometryBuffer, std::function<char*(size_t)> binningBuffer,
      std::function<char*(size_t)> imageBuffer, const int P, int D, int M, int E, const int width, int height,
      const float* means3D, const float* shs, const float* colors_precomp, const float* opacities, const float* scales,
      const float scale_modifier, const float* rotations, const float* cov3D_precomp, const float* viewmatrix,
      const float* projmatrix, const float* cam_pos, const float tan_fovx, float tan_fovy, const bool prefiltered,
      float* out_color, float* out_opacity, const float* extra = nullptr, float* out_extra = nullptr,
      int* radii = nullptr, bool debug = false, bool colmap = false);

  static void backward(const int P, int D, int M, int R, int E, const int width, int height, const float* means3D,
      const float* shs, const float* colors_precomp, const float* scales, const float scale_modifier,
      const float* rotations, const float* cov3D_precomp, const float* viewmatrix, const float* projmatrix,
      const float* campos, const float* extra, const float tan_fovx, float tan_fovy, const int* radii,
      char* geom_buffer, char* binning_buffer, char* img_buffer, const float* out_opacity, const float* dL_dpix,
      const float* dL_dout_opacity, const float* dL_dout_extra, float* dL_dmean2D, float* dL_dconic, float* dL_dopacity,
      float* dL_dcolor, float* dL_dmean3D, float* dL_dcov3D, float* dL_dsh, float* dL_dscale, float* dL_drot,
      float* dL_dextra = nullptr, bool debug = false, bool colmap = false);
};

template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment) {
  std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
  ptr                = reinterpret_cast<T*>(offset);
  chunk              = reinterpret_cast<char*>(ptr + count);
}

struct GeometryState {
  size_t scan_size;
  float* depths;
  char* scanning_space;
  bool* clamped;
  int* internal_radii;
  float2* means2D;
  float* cov3D;
  float4* conic_opacity;
  float* rgb;
  uint32_t* point_offsets;
  uint32_t* tiles_touched;

  static GeometryState fromChunk(char*& chunk, size_t P);
};

struct ImageState {
  uint2* ranges;
  uint32_t* n_contrib;
  // float* accum_alpha;

  static ImageState fromChunk(char*& chunk, size_t N);
};

struct BinningState {
  size_t sorting_size;
  uint64_t* point_list_keys_unsorted;
  uint64_t* point_list_keys;
  uint32_t* point_list_unsorted;
  uint32_t* point_list;
  char* list_sorting_space;

  static BinningState fromChunk(char*& chunk, size_t P);
};

template <typename T>
size_t required(size_t P) {
  char* size = nullptr;
  T::fromChunk(size, P);
  return ((size_t) size) + 128;
}
};  // namespace GaussianRasterizer
