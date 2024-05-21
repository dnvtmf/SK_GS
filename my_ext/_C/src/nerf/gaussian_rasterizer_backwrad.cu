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

#include "gaussian_render.h"
#include "util.cuh"

namespace cg = cooperative_groups;

namespace GaussianRasterizer {
// Backward pass for conversion of spherical harmonics to RGB for each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos,
    const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs) {
  // Compute intermediate values, as it is done during forward
  glm::vec3 pos      = means[idx];
  glm::vec3 dir_orig = pos - campos;
  glm::vec3 dir      = dir_orig / glm::length(dir_orig);

  glm::vec3* sh = ((glm::vec3*) shs) + idx * max_coeffs;

  // Use PyTorch rule for clamping: if clamping was applied, gradient becomes 0.
  glm::vec3 dL_dRGB = dL_dcolor[idx];
  dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
  dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
  dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

  glm::vec3 dRGBdx(0, 0, 0);
  glm::vec3 dRGBdy(0, 0, 0);
  glm::vec3 dRGBdz(0, 0, 0);
  float x = dir.x;
  float y = dir.y;
  float z = dir.z;

  // Target location for this Gaussian to write SH gradients to
  glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

  // No tricks here, just high school-level calculus.
  float dRGBdsh0 = SH_C0;
  dL_dsh[0]      = dRGBdsh0 * dL_dRGB;
  if (deg > 0) {
    float dRGBdsh1 = -SH_C1 * y;
    float dRGBdsh2 = SH_C1 * z;
    float dRGBdsh3 = -SH_C1 * x;
    dL_dsh[1]      = dRGBdsh1 * dL_dRGB;
    dL_dsh[2]      = dRGBdsh2 * dL_dRGB;
    dL_dsh[3]      = dRGBdsh3 * dL_dRGB;

    dRGBdx = -SH_C1 * sh[3];
    dRGBdy = -SH_C1 * sh[1];
    dRGBdz = SH_C1 * sh[2];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;

      float dRGBdsh4 = SH_C2[0] * xy;
      float dRGBdsh5 = SH_C2[1] * yz;
      float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
      float dRGBdsh7 = SH_C2[3] * xz;
      float dRGBdsh8 = SH_C2[4] * (xx - yy);
      dL_dsh[4]      = dRGBdsh4 * dL_dRGB;
      dL_dsh[5]      = dRGBdsh5 * dL_dRGB;
      dL_dsh[6]      = dRGBdsh6 * dL_dRGB;
      dL_dsh[7]      = dRGBdsh7 * dL_dRGB;
      dL_dsh[8]      = dRGBdsh8 * dL_dRGB;

      dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
      dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
      dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

      if (deg > 2) {
        float dRGBdsh9  = SH_C3[0] * y * (3.f * xx - yy);
        float dRGBdsh10 = SH_C3[1] * xy * z;
        float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
        float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
        dL_dsh[9]       = dRGBdsh9 * dL_dRGB;
        dL_dsh[10]      = dRGBdsh10 * dL_dRGB;
        dL_dsh[11]      = dRGBdsh11 * dL_dRGB;
        dL_dsh[12]      = dRGBdsh12 * dL_dRGB;
        dL_dsh[13]      = dRGBdsh13 * dL_dRGB;
        dL_dsh[14]      = dRGBdsh14 * dL_dRGB;
        dL_dsh[15]      = dRGBdsh15 * dL_dRGB;

        dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy + SH_C3[1] * sh[10] * yz + SH_C3[2] * sh[11] * -2.f * xy +
                   SH_C3[3] * sh[12] * -3.f * 2.f * xz + SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
                   SH_C3[5] * sh[14] * 2.f * xz + SH_C3[6] * sh[15] * 3.f * (xx - yy));

        dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) + SH_C3[1] * sh[10] * xz +
                   SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) + SH_C3[3] * sh[12] * -3.f * 2.f * yz +
                   SH_C3[4] * sh[13] * -2.f * xy + SH_C3[5] * sh[14] * -2.f * yz + SH_C3[6] * sh[15] * -3.f * 2.f * xy);

        dRGBdz += (SH_C3[1] * sh[10] * xy + SH_C3[2] * sh[11] * 4.f * 2.f * yz +
                   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) + SH_C3[4] * sh[13] * 4.f * 2.f * xz +
                   SH_C3[5] * sh[14] * (xx - yy));
      }
    }
  }

  // The view direction is an input to the computation.
  // View direction is influenced by the Gaussian's mean, so SHs gradients must propagate back into 3D position.
  glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

  // Account for normalization of direction
  float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

  // Gradients of loss w.r.t. Gaussian means, but only the portion
  // that is caused because the mean affects the view-dependent color.
  // Additional mean gradient is accumulated in below methods.
  dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

void preprocess_backward_colmap(int P, int D, int M, const float3* means3D, const int* radii, const float* shs,
    const bool* clamped, const glm::vec3* scales, const glm::vec4* rotations, const float scale_modifier,
    const float* cov3Ds, const float* viewmatrix, const float* projmatrix, const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy, const glm::vec3* campos, const float3* dL_dmean2D, const float* dL_dconic,
    glm::vec3* dL_dmean3D, float* dL_dcolor, float* dL_dcov3D, float* dL_dsh, glm::vec3* dL_dscale, glm::vec4* dL_drot);

void preprocess_backward(int P, int D, int M, const float3* means3D, const int* radii, const float* shs,
    const bool* clamped, const float3* scales, const float4* rotations, const float scale_modifier, const float* cov3Ds,
    const float* viewmatrix, const float* projmatrix, const float focal_x, float focal_y, const float tan_fovx,
    float tan_fovy, const glm::vec3* campos, const float3* dL_dmean2D, const float* dL_dconic, glm::vec3* dL_dmean3D,
    float* dL_dcolor, float* dL_dcov3D, float* dL_dsh, float3* dL_dscale, float4* dL_drot);

void render_backward(int P, int W, int H, int E, const dim3 grid, const dim3 block, const uint2* ranges,
    const uint32_t* point_list, /*const float* bg_color,*/ const float2* means2D, const float4* conic_opacity,
    const float* colors, const float* extras, const float* out_opacity, const uint32_t* n_contrib,
    const float* dL_dpixels, const float* dL_dout_extras, const float* dL_dout_opacity, float3* dL_dmean2D,
    float4* dL_dconic2D, float* dL_dopacity, float* dL_dcolors, float* dL_dextras);

// Produce necessary gradients for optimization, corresponding to forward render pass
void Rasterizer::backward(const int P, int D, int M, int R, int E, const int width, int height, const float* means3D,
    const float* shs, const float* colors_precomp, const float* scales, const float scale_modifier,
    const float* rotations, const float* cov3D_precomp, const float* viewmatrix, const float* projmatrix,
    const float* campos, const float* extra, const float tan_fovx, float tan_fovy, const int* radii, char* geom_buffer,
    char* binning_buffer, char* img_buffer, const float* out_opacity, const float* dL_dpix,
    const float* dL_dout_opacity, const float* dL_dout_extra, float* dL_dmean2D, float* dL_dconic, float* dL_dopacity,
    float* dL_dcolor, float* dL_dmean3D, float* dL_dcov3D, float* dL_dsh, float* dL_dscale, float* dL_drot,
    float* dL_dextra, bool debug, bool colmap) {
  GeometryState geomState   = GeometryState::fromChunk(geom_buffer, P);
  BinningState binningState = BinningState::fromChunk(binning_buffer, R);
  ImageState imgState       = ImageState::fromChunk(img_buffer, width * height);

  if (radii == nullptr) {
    radii = geomState.internal_radii;
  }

  const float focal_y = height / (2.0f * tan_fovy);
  const float focal_x = width / (2.0f * tan_fovx);

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);
  // if (debug) cudaDeviceSynchronize();
  // printf("before render_backward");
  // Compute loss gradients w.r.t. 2D mean position, conic matrix,
  // opacity and RGB of Gaussians from per-pixel loss gradients.
  // If we were given precomputed colors and not SHs, use them.
  const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
  render_backward(P, width, height, E, tile_grid, block, imgState.ranges, binningState.point_list, geomState.means2D,
      geomState.conic_opacity, color_ptr, extra, out_opacity, imgState.n_contrib, dL_dpix, dL_dout_extra,
      dL_dout_opacity, (float3*) dL_dmean2D, (float4*) dL_dconic, dL_dopacity, dL_dcolor, dL_dextra);
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("render");
  // printf("after render_backward");

  // Take care of the rest of preprocessing. Was the precomputed covariance given to us or a scales/rot pair?
  // If precomputed, pass that. If not, use the one we computed ourselves.
  const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
  if (colmap) {
    preprocess_backward_colmap(P, D, M, (float3*) means3D, radii, shs, geomState.clamped, (glm::vec3*) scales,
        (glm::vec4*) rotations, scale_modifier, cov3D_ptr, viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy,
        (glm::vec3*) campos, (float3*) dL_dmean2D, dL_dconic, (glm::vec3*) dL_dmean3D, dL_dcolor, dL_dcov3D, dL_dsh,
        (glm::vec3*) dL_dscale, (glm::vec4*) dL_drot);
  } else {
    preprocess_backward(P, D, M, (float3*) means3D, radii, shs, geomState.clamped, (float3*) scales,
        (float4*) rotations, scale_modifier, cov3D_ptr, viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy,
        (glm::vec3*) campos, (float3*) dL_dmean2D, dL_dconic, (glm::vec3*) dL_dmean3D, dL_dcolor, dL_dcov3D, dL_dsh,
        (float3*) dL_dscale, (float4*) dL_drot);
  }
  if (debug) cudaDeviceSynchronize();
  CHECK_CUDA_ERROR("preprocess");
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, at::optional<Tensor>>
    RasterizeGaussiansBackwardCUDA(
        // scalar parameters
        const float scale_modifier, const float tan_fovx, const float tan_fovy, const int degree, const bool debug,
        const bool colmap,
        // tensor parameters
        const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix, const torch::Tensor& campos,
        // inputs
        const torch::Tensor& means3D, const torch::Tensor& colors, const at::optional<Tensor> extras,
        const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& cov3D_precomp,
        const torch::Tensor& sh,
        // outputs
        const int R, const torch::Tensor& radii, const Tensor& out_opacity,
        // grad_outputs
        const torch::Tensor& dL_dout_color, const Tensor& dL_dout_opacity, const at::optional<Tensor>& dL_dout_extra,
        // grad_inputs
        torch::optional<Tensor>& grad_means2D, torch::optional<Tensor>& grad_conic,
        torch::optional<Tensor>& grad_opacity,
        // buffer
        const torch::Tensor& geomBuffer, const torch::Tensor& binningBuffer, const torch::Tensor& imageBuffer) {
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  const int E = (extras.has_value() && dL_dout_extra.has_value()) ? extras.value().size(-1) : 0;

  int M = 0;
  if (sh.size(0) != 0) {
    M = sh.size(1);
  }

  Tensor dL_dmeans3D   = torch::zeros({P, 3}, means3D.options());
  Tensor dL_dmeans2D   = grad_means2D.has_value() ? grad_means2D.value() : torch::zeros({P, 3}, means3D.options());
  Tensor dL_dconic     = grad_conic.has_value() ? grad_conic.value() : torch::zeros({P, 2, 2}, means3D.options());
  Tensor dL_dopacity   = grad_opacity.has_value() ? grad_opacity.value() : torch::zeros({P, 1}, means3D.options());
  Tensor dL_dsh        = torch::zeros({P, M, 3}, means3D.options());
  Tensor dL_dscales    = torch::zeros({P, 3}, means3D.options());
  Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  Tensor dL_dcolors    = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  Tensor dL_dcov3D     = torch::zeros({P, 6}, means3D.options());
  at::optional<Tensor> dL_dextras;
  if (E > 0) dL_dextras = torch::zeros({P, E}, means3D.options());

  if (P != 0) {
    Rasterizer::backward(P, degree, M, R, E, W, H, means3D.contiguous().data<float>(), sh.contiguous().data<float>(),
        colors.contiguous().data<float>(), scales.data_ptr<float>(), scale_modifier, rotations.data_ptr<float>(),
        cov3D_precomp.contiguous().data<float>(), viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(), campos.contiguous().data<float>(),
        E > 0 ? extras.value().contiguous().data<float>() : nullptr, tan_fovx, tan_fovy, radii.contiguous().data<int>(),
        reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()), out_opacity.contiguous().data<float>(),
        dL_dout_color.contiguous().data<float>(), dL_dout_opacity.contiguous().data<float>(),
        E > 0 ? dL_dout_extra.value().contiguous().data<float>() : nullptr, dL_dmeans2D.contiguous().data<float>(),
        dL_dconic.contiguous().data<float>(), dL_dopacity.contiguous().data<float>(),
        dL_dcolors.contiguous().data<float>(), dL_dmeans3D.contiguous().data<float>(),
        dL_dcov3D.contiguous().data<float>(), dL_dsh.contiguous().data<float>(), dL_dscales.contiguous().data<float>(),
        dL_drotations.contiguous().data<float>(), E > 0 ? dL_dextras.value().data<float>() : nullptr, debug, colmap);
  }

  return std::make_tuple(
      dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dextras);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> gaussian_rasterize_perpare_backward(
    // scalar parameters
    const float scale_modifier, const float tan_fovx, const float tan_fovy, const int degree, const bool colmap,
    // tensor parameters
    const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix, const torch::Tensor& campos,
    // inputs
    const torch::Tensor& means3D, const torch::Tensor& colors, const torch::Tensor& scales,
    const torch::Tensor& rotations, const torch::Tensor& cov3D_precomp, const torch::Tensor& sh,
    // outputs
    const int R, const torch::Tensor& radii, const Tensor& out_opacity,
    // grad_outputs
    const torch::Tensor& dL_dout_color, const Tensor& dL_dout_opacity,
    // grad_inputs
    torch::optional<Tensor>& grad_means2D, torch::optional<Tensor>& grad_conic, torch::optional<Tensor>& grad_opacity,
    // buffer
    const torch::Tensor& geomBuffer, const torch::Tensor& binningBuffer, const torch::Tensor& imageBuffer) {
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if (sh.size(0) != 0) {
    M = sh.size(1);
  }

  Tensor dL_dmeans3D   = torch::zeros({P, 3}, means3D.options());
  Tensor dL_dmeans2D   = grad_means2D.has_value() ? grad_means2D.value() : torch::zeros({P, 3}, means3D.options());
  Tensor dL_dconic     = grad_conic.has_value() ? grad_conic.value() : torch::zeros({P, 2, 2}, means3D.options());
  Tensor dL_dopacity   = grad_opacity.has_value() ? grad_opacity.value() : torch::zeros({P, 1}, means3D.options());
  Tensor dL_dcolors    = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  Tensor dL_dcov3D     = torch::zeros({P, 6}, means3D.options());
  Tensor dL_dsh        = torch::zeros({P, M, 3}, means3D.options());
  Tensor dL_dscales    = torch::zeros({P, 3}, means3D.options());
  Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  if (P != 0) {
    char* geom_buffer = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
    // char* binning_buffer      = reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr());
    // char* img_buffer          = reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr());
    GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
    // BinningState binningState = BinningState::fromChunk(binning_buffer, R);
    // ImageState imgState       = ImageState::fromChunk(img_buffer, W * H);

    const int* radii_ptr = radii.contiguous().data_ptr<int>();
    if (radii_ptr == nullptr) radii_ptr = geomState.internal_radii;
    const float focal_y = H / (2.0f * tan_fovy);
    const float focal_x = W / (2.0f * tan_fovx);

    // const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    // const dim3 block(BLOCK_X, BLOCK_Y, 1);
    // Take care of the rest of preprocessing.
    // Was the precomputed covariance given to us or a scales/rot pair?
    // If precomputed, pass that. If not, use the one we computed ourselves.
    const float* cov3D_ptr = cov3D_precomp.contiguous().data<float>();
    cov3D_ptr              = (cov3D_ptr != nullptr) ? cov3D_ptr : geomState.cov3D;
    preprocess_backward(P, degree, M, (float3*) means3D.contiguous().data<float>(), radii_ptr,
        sh.contiguous().data<float>(), geomState.clamped, (float3*) scales.contiguous().data<float>(),
        (float4*) rotations.data_ptr<float>(), scale_modifier, cov3D_ptr, viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(), focal_x, focal_y, tan_fovx, tan_fovy,
        (glm::vec3*) campos.contiguous().data<float>(), (float3*) dL_dmeans2D.contiguous().data<float>(),
        dL_dconic.contiguous().data<float>(), (glm::vec3*) dL_dmeans3D.contiguous().data<float>(),
        dL_dcolors.contiguous().data<float>(), dL_dcov3D.contiguous().data<float>(), dL_dsh.contiguous().data<float>(),
        (float3*) dL_dscales.contiguous().data<float>(), (float4*) dL_drotations.contiguous().data<float>());
    CHECK_CUDA_ERROR("preprocess");
  }

  return std::make_tuple(
      dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

REGIST_PYTORCH_EXTENSION(
    nerf_gaussian_render_backward, { m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA); })
}  // namespace GaussianRasterizer