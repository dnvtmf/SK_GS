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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "gaussian_render.h"
#include "util.cuh"

namespace cg = cooperative_groups;

namespace GaussianRasterizer {

// Helper function to find the next-highest bit of the MSB on the CPU.
uint32_t getHigherMsb(uint32_t n) {
  uint32_t msb  = sizeof(n) * 4;
  uint32_t step = msb;
  while (step > 1) {
    step /= 2;
    if (n >> msb)
      msb += step;
    else
      msb -= step;
  }
  if (n >> msb) msb++;
  return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps. Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(int P, const float2* points_xy, const float* depths, const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted, int* radii, dim3 grid) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Generate no key/value pair for invisible Gaussians
  if (radii[idx] > 0) {
    // Find this Gaussian's offset in buffer for writing keys/values.
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    uint2 rect_min, rect_max;

    getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

    // For each tile that the bounding rect overlaps, emit a key/value pair.
    // The key is |  tile ID  |      depth      |, and the value is the ID of the Gaussian.
    // Sorting the values with this key yields Gaussian IDs in a list,
    // such that they are first sorted by tile and then by depth.
    for (int y = rect_min.y; y < rect_max.y; y++) {
      for (int x = rect_min.x; x < rect_max.x; x++) {
        uint64_t key = y * grid.x + x;
        key <<= 32;
        key |= *((uint32_t*) &depths[idx]);
        gaussian_keys_unsorted[off]   = key;
        gaussian_values_unsorted[off] = idx;
        off++;
      }
    }
  }
}

// Check keys to see if it is at the start/end of one tile's range in the full sorted list.
// If yes, write start/end of this tile. Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= L) return;

  // Read tile ID from key. Update start/end of tile range if at limit.
  uint64_t key      = point_list_keys[idx];
  uint32_t currtile = key >> 32;
  if (idx == 0)
    ranges[currtile].x = 0;
  else {
    uint32_t prevtile = point_list_keys[idx - 1] >> 32;
    if (currtile != prevtile) {
      ranges[prevtile].y = idx;
      ranges[currtile].x = idx;
    }
  }
  if (idx == L - 1) ranges[currtile].y = L;
}

// Forward method for converting the input spherical harmonics coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(
    int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for Efficient View Synthesis" by Zhang et al. (2022)
  glm::vec3 pos = means[idx];
  glm::vec3 dir = pos - campos;
  dir           = dir / glm::length(dir);

  glm::vec3* sh    = ((glm::vec3*) shs) + idx * max_coeffs;
  glm::vec3 result = SH_C0 * sh[0];

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result  = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] + SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
               SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] + SH_C3[1] * xy * z * sh[10] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] + SH_C3[5] * z * (xx - yy) * sh[14] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
      }
    }
  }
  result += 0.5f;

  // RGB colors are clamped to positive values.
  // If values are clamped, we need to keep track of this for the backward pass.
  clamped[3 * idx + 0] = (result.x < 0);
  clamped[3 * idx + 1] = (result.y < 0);
  clamped[3 * idx + 2] = (result.z < 0);
  return glm::max(result, 0.0f);
}

void render_forward(const dim3 grid, dim3 block, const uint2* ranges, const uint32_t* point_list, int W, int H, int E,
    const float2* means2D, const float* colors, const float4* conic_opacity, const float* extra, uint32_t* n_contrib,
    /*const float* bg_color,*/ float* out_color, float* out_opacity, float* out_extra);
void preprocess_forward_colmap(int P, int D, int M, const float* means3D, const glm::vec3* scales,
    const float scale_modifier, const glm::vec4* rotations, const float* opacities, const float* shs, bool* clamped,
    const float* cov3D_precomp, const float* colors_precomp, const float* viewmatrix, const float* projmatrix,
    const glm::vec3* cam_pos, const int W, int H, const float focal_x, float focal_y, const float tan_fovx,
    float tan_fovy, int* radii, float2* means2D, float* depths, float* cov3Ds, float* rgb, float4* conic_opacity,
    const dim3 grid, uint32_t* tiles_touched, bool prefiltered);

void preprocess_forward(int P, int D, int M, const float* means3D, const float3* scales, const float scale_modifier,
    const float4* rotations, const float* opacities, const float* shs, bool* clamped, const float* cov3D_precomp,
    const float* colors_precomp, const float* viewmatrix, const float* projmatrix, const glm::vec3* cam_pos,
    const int W, int H, const float focal_x, float focal_y, const float tan_fovx, float tan_fovy, int* radii,
    float2* means2D, float* depths, float* cov3Ds, float* rgb, float4* conic_opacity, const dim3 grid,
    uint32_t* tiles_touched, bool prefiltered);

// Forward rendering procedure for differentiable rasterization of Gaussians.
int Rasterizer::forward(std::function<char*(size_t)> geometryBuffer, std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> imageBuffer, const int P, int D, int M, int E, const int width, int height,
    const float* means3D, const float* shs, const float* colors_precomp, const float* opacities, const float* scales,
    const float scale_modifier, const float* rotations, const float* cov3D_precomp, const float* viewmatrix,
    const float* projmatrix, const float* cam_pos, const float tan_fovx, float tan_fovy, const bool prefiltered,
    float* out_color, float* out_opacity, const float* extra, float* out_extra, int* radii, bool debug, bool colmap) {
  const float focal_y = height / (2.0f * tan_fovy);
  const float focal_x = width / (2.0f * tan_fovx);

  size_t chunk_size       = required<GeometryState>(P);
  char* chunkptr          = geometryBuffer(chunk_size);
  GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

  if (radii == nullptr) {
    radii = geomState.internal_radii;
  }

  dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  dim3 block(BLOCK_X, BLOCK_Y, 1);

  // Dynamically resize image-based auxiliary buffers during training
  size_t img_chunk_size = required<ImageState>(width * height);
  char* img_chunkptr    = imageBuffer(img_chunk_size);
  ImageState imgState   = ImageState::fromChunk(img_chunkptr, width * height);

  if (NUM_CHANNELS != 3 && colors_precomp == nullptr) {
    throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
  }

  // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
  if (colmap) {
    preprocess_forward_colmap(P, D, M, means3D, (glm::vec3*) scales, scale_modifier, (glm::vec4*) rotations, opacities,
        shs, geomState.clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix, (glm::vec3*) cam_pos, width,
        height, focal_x, focal_y, tan_fovx, tan_fovy, radii, geomState.means2D, geomState.depths, geomState.cov3D,
        geomState.rgb, geomState.conic_opacity, tile_grid, geomState.tiles_touched, prefiltered);
  } else {
    preprocess_forward(P, D, M, means3D, (float3*) scales, scale_modifier, (float4*) rotations, opacities, shs,
        geomState.clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix, (glm::vec3*) cam_pos, width, height,
        focal_x, focal_y, tan_fovx, tan_fovy, radii, geomState.means2D, geomState.depths, geomState.cov3D,
        geomState.rgb, geomState.conic_opacity, tile_grid, geomState.tiles_touched, prefiltered);
  }
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("preprocess");

  // Compute prefix sum over full list of touched tile counts by Gaussians
  // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
  cub::DeviceScan::InclusiveSum(
      geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P);
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("InclusiveSum");
  // Retrieve total number of Gaussian instances to launch and resize aux buffers
  int num_rendered;
  cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("cudaMemcpy");

  size_t binning_chunk_size = required<BinningState>(num_rendered);
  char* binning_chunkptr    = binningBuffer(binning_chunk_size);
  BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

  // For each instance to be rendered, produce adequate [ tile | depth ] key
  // and corresponding dublicated Gaussian indices to be sorted
  duplicateWithKeys KERNEL_ARG((P + 255) / 256, 256)(P, geomState.means2D, geomState.depths, geomState.point_offsets,
      binningState.point_list_keys_unsorted, binningState.point_list_unsorted, radii, tile_grid);
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("duplicateWithKeys");

  int bit = getHigherMsb(tile_grid.x * tile_grid.y);

  // Sort complete list of (duplicated) Gaussian indices by keys
  cub::DeviceRadixSort::SortPairs(binningState.list_sorting_space, binningState.sorting_size,
      binningState.point_list_keys_unsorted, binningState.point_list_keys, binningState.point_list_unsorted,
      binningState.point_list, num_rendered, 0, 32 + bit);
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("SortPairs");
  cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("cudaMemset");
  // Identify start and end of per-tile workloads in sorted list
  if (num_rendered > 0) {
    identifyTileRanges KERNEL_ARG((num_rendered + 255) / 256, 256)(
        num_rendered, binningState.point_list_keys, imgState.ranges);
    if (debug) cudaDeviceSynchronize();
    CHECK_CUDA_ERROR("identifyTileRanges");
  }

  // Let each tile blend its range of Gaussians independently in parallel
  const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

  render_forward(tile_grid, block, imgState.ranges, binningState.point_list, width, height, E, geomState.means2D,
      feature_ptr, geomState.conic_opacity, extra, imgState.n_contrib, out_color, out_opacity, out_extra);

  return num_rendered;
}

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
  auto lambda = [&t](size_t N) {
    t.resize_({(long long) N});
    return reinterpret_cast<char*>(t.contiguous().data_ptr());
  };
  return lambda;
}

std::tuple<int, torch::Tensor, Tensor, Tensor, torch::Tensor, torch::Tensor, torch::Tensor, Tensor>
    RasterizeGaussiansCUDA(
        // const params
        const int image_height, const int image_width, const float tan_fovx, const float tan_fovy, const int degree,
        const float scale_modifier, const bool prefiltered, const bool debug, const bool colmap,
        // tenser params
        const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix, const torch::Tensor& campos,
        // inputs
        const torch::Tensor& means3D, const torch::Tensor& opacity, const torch::Tensor& sh,
        const torch::Tensor& scales, const torch::Tensor& rotations, const at::optional<Tensor>& extras,
        const torch::Tensor& colors, const torch::Tensor& cov3D_precomp) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  const int E = extras.has_value() ? extras.value().size(-1) : 0;

  // auto int_opts   = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color   = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_opacity = torch::full({H, W}, 0.0, float_opts);
  torch::Tensor radii       = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  Tensor out_extras;
  if (extras.has_value()) out_extras = torch::zeros({E, H, W}, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer                 = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer              = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer                  = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc    = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc     = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0) {
    int M = 0;
    if (sh.size(0) != 0) {
      M = sh.size(1);
    }

    rendered = Rasterizer::forward(geomFunc, binningFunc, imgFunc, P, degree, M, E, W, H,
        means3D.contiguous().data<float>(), sh.contiguous().data_ptr<float>(), colors.contiguous().data<float>(),
        opacity.contiguous().data<float>(), scales.contiguous().data_ptr<float>(), scale_modifier,
        rotations.contiguous().data_ptr<float>(), cov3D_precomp.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(), projmatrix.contiguous().data<float>(), campos.contiguous().data<float>(),
        tan_fovx, tan_fovy, prefiltered, out_color.contiguous().data<float>(), out_opacity.data<float>(),
        extras.has_value() ? extras.value().contiguous().data<float>() : nullptr,
        extras.has_value() ? out_extras.data<float>() : nullptr, radii.contiguous().data<int>(), debug, colmap);
  }
  return std::make_tuple(rendered, out_color, out_opacity, radii, geomBuffer, binningBuffer, imgBuffer, out_extras);
}

REGIST_PYTORCH_EXTENSION(nerf_gaussian_render_forward, { m.def("rasterize_gaussians", &RasterizeGaussiansCUDA); })
}  // namespace GaussianRasterizer