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

#include <cuda.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <fstream>
#include <iostream>
#include <numeric>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <glm/glm.hpp>

#include "gaussian_render.h"
#include "util.cuh"

namespace cg = cooperative_groups;
namespace GaussianRasterizer {

GeometryState GeometryState::fromChunk(char*& chunk, size_t P) {
  GeometryState geom;
  obtain(chunk, geom.depths, P, 128);
  obtain(chunk, geom.clamped, P * 3, 128);
  obtain(chunk, geom.internal_radii, P, 128);
  obtain(chunk, geom.means2D, P, 128);
  obtain(chunk, geom.cov3D, P * 6, 128);
  obtain(chunk, geom.conic_opacity, P, 128);
  obtain(chunk, geom.rgb, P * 3, 128);
  obtain(chunk, geom.tiles_touched, P, 128);
  cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
  obtain(chunk, geom.scanning_space, geom.scan_size, 128);
  obtain(chunk, geom.point_offsets, P, 128);
  return geom;
}

ImageState ImageState::fromChunk(char*& chunk, size_t N) {
  ImageState img;
  // obtain(chunk, img.accum_alpha, N, 128);
  obtain(chunk, img.n_contrib, N, 128);
  obtain(chunk, img.ranges, N, 128);
  return img;
}

BinningState BinningState::fromChunk(char*& chunk, size_t P) {
  BinningState binning;
  obtain(chunk, binning.point_list, P, 128);
  obtain(chunk, binning.point_list_unsorted, P, 128);
  obtain(chunk, binning.point_list_keys, P, 128);
  obtain(chunk, binning.point_list_keys_unsorted, P, 128);
  cub::DeviceRadixSort::SortPairs(nullptr, binning.sorting_size, binning.point_list_keys_unsorted,
      binning.point_list_keys, binning.point_list_unsorted, binning.point_list, P);
  obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
  return binning;
}

/*
// Wrapper method to call auxiliary coarse frustum containment test. Mark all Gaussians that pass it.
__global__ void checkFrustum(
    int P, const float* orig_points, const float* viewmatrix, const float* projmatrix, bool* present) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  float3 p_view;
  present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}
// Mark Gaussians as visible/invisible, based on view frustum testing
void Rasterizer::markVisible(int P, float* means3D, float* viewmatrix, float* projmatrix, bool* present) {
  checkFrustum KERNEL_ARG((P + 255) / 256, 256)(P, means3D, viewmatrix, projmatrix, present);
}
torch::Tensor markVisible(torch::Tensor& means3D, torch::Tensor& viewmatrix, torch::Tensor& projmatrix) {
  const int P = means3D.size(0);

  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

  if (P != 0) {
    Rasterizer::markVisible(P, means3D.contiguous().data<float>(), viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(), present.contiguous().data<bool>());
  }

  return present;
}

REGIST_PYTORCH_EXTENSION(nerf_gaussian_render_imp, { m.def("mark_visible", &markVisible); })
*/
}  // namespace GaussianRasterizer