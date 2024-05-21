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

__forceinline__ __device__ float ndc2Pix(float v, int S) { return ((v + 1.0) * S - 1.0) * 0.5; }

__forceinline__ __device__ float3 transformPoint4x3_colmap(const float3& p, const float* matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
  };
  return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4_colmap(const float3& p, const float* matrix) {
  float4 transformed = {matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
      matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
  return transformed;
}

__forceinline__ __device__ float3 transformVec4x3_colmap(const float3& p, const float* matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose_colmap(const float3& p, const float* matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ bool in_frustum_colmap(int idx, const float* orig_points, const float* viewmatrix,
    const float* projmatrix, bool prefiltered, float3& p_view) {
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};

  // Bring points to screen space
  float4 p_hom = transformPoint4x4_colmap(p_orig, projmatrix);
  // float p_w    = 1.0f / (p_hom.w + 0.0000001f);
  // float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
  p_view = transformPoint4x3_colmap(p_orig, viewmatrix);

  if (p_view.z <= 0.2f)  // || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
  {
    if (prefiltered) {
      printf("Point is filtered although prefiltered is set. This shouldn't happen!");
      __trap();
    }
    return false;
  }
  return true;
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D_colmap(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy,
    const float* cov3D, const float* viewmatrix) {
  // The following models the steps outlined by equations 29 and 31 in "EWA Splatting" (Zwicker et al., 2002).
  // Additionally considers aspect / scaling of viewport.
  // Transposes used to account for row-/column-major conventions.
  float3 t = transformPoint4x3_colmap(mean, viewmatrix);

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x              = min(limx, max(-limx, txtz)) * t.z;
  t.y              = min(limy, max(-limy, tytz)) * t.z;

  glm::mat3 J = glm::mat3(focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z), 0.0f, focal_y / t.z,
      -(focal_y * t.y) / (t.z * t.z), 0, 0, 0);

  glm::mat3 W = glm::mat3(viewmatrix[0], viewmatrix[4], viewmatrix[8], viewmatrix[1], viewmatrix[5], viewmatrix[9],
      viewmatrix[2], viewmatrix[6], viewmatrix[10]);

  glm::mat3 T = W * J;

  glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

  // Apply low-pass filter: every Gaussian should be at least
  // one pixel wide/high. Discard 3rd row and column.
  cov[0][0] += 0.3f;
  cov[1][1] += 0.3f;
  return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space.
// Also takes care of quaternion normalization.
__device__ void computeCov3D_colmap(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D) {
  // Create scaling matrix
  glm::mat3 S = glm::mat3(1.0f);
  S[0][0]     = mod * scale.x;
  S[1][1]     = mod * scale.y;
  S[2][2]     = mod * scale.z;

  // Normalize quaternion to get valid rotation
  glm::vec4 q = rot;  // / glm::length(rot);
  float x     = q.x;
  float y     = q.y;
  float z     = q.z;
  float r     = q.w;

  // Compute rotation matrix from quaternion
  glm::mat3 R = glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
      2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x), 2.f * (x * z - r * y),
      2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

  glm::mat3 M = S * R;

  // Compute 3D world covariance matrix Sigma
  glm::mat3 Sigma = glm::transpose(M) * M;

  // Covariance is symmetric, only store upper right
  cov3D[0] = Sigma[0][0];
  cov3D[1] = Sigma[0][1];
  cov3D[2] = Sigma[0][2];
  cov3D[3] = Sigma[1][1];
  cov3D[4] = Sigma[1][2];
  cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA_colmap(int P, int D, int M, const float* orig_points, const glm::vec3* scales,
    const float scale_modifier, const glm::vec4* rotations, const float* opacities, const float* shs, bool* clamped,
    const float* cov3D_precomp, const float* colors_precomp, const float* viewmatrix, const float* projmatrix,
    const glm::vec3* cam_pos, const int W, int H, const float tan_fovx, float tan_fovy, const float focal_x,
    float focal_y, int* radii, float2* points_xy_image, float* depths, float* cov3Ds, float* rgb, float4* conic_opacity,
    const dim3 grid, uint32_t* tiles_touched, bool prefiltered) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Initialize radius and touched tiles to 0. If this isn't changed, this Gaussian will not be processed further.
  radii[idx]         = 0;
  tiles_touched[idx] = 0;

  // Perform near culling, quit if outside.
  float3 p_view;
  if (!in_frustum_colmap(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) return;

  // Transform point by projecting
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
  float4 p_hom  = transformPoint4x4_colmap(p_orig, projmatrix);
  float p_w     = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

  // If 3D covariance matrix is precomputed, use it, otherwise compute from scaling and rotation parameters.
  const float* cov3D;
  if (cov3D_precomp != nullptr) {
    cov3D = cov3D_precomp + idx * 6;
  } else {
    computeCov3D_colmap(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
    cov3D = cov3Ds + idx * 6;
  }

  // Compute 2D screen-space covariance matrix
  float3 cov = computeCov2D_colmap(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

  // Invert covariance (EWA algorithm)
  float det = (cov.x * cov.z - cov.y * cov.y);
  if (det == 0.0f) return;
  float det_inv = 1.f / det;
  float3 conic  = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

  // Compute extent in screen space (by finding eigenvalues of 2D covariance matrix).
  // Use extent to compute a bounding rectangle of screen-space tiles that this Gaussian overlaps with.
  // Quit if rectangle covers 0 tiles.
  float mid          = 0.5f * (cov.x + cov.z);
  float lambda1      = mid + sqrt(max(0.1f, mid * mid - det));
  float lambda2      = mid - sqrt(max(0.1f, mid * mid - det));
  float my_radius    = ceil(3.f * sqrt(max(lambda1, lambda2)));
  float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
  uint2 rect_min, rect_max;
  getRect(point_image, my_radius, rect_min, rect_max, grid);
  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

  // If colors have been precomputed, use them, otherwise convert spherical harmonics coefficients to RGB color.
  if (colors_precomp == nullptr) {
    glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*) orig_points, *cam_pos, shs, clamped);
    rgb[idx * C + 0] = result.x;
    rgb[idx * C + 1] = result.y;
    rgb[idx * C + 2] = result.z;
  }

  // Store some useful helper data for the next steps.
  depths[idx]          = p_view.z;
  radii[idx]           = my_radius;
  points_xy_image[idx] = point_image;
  // Inverse 2D covariance and opacity neatly pack into one float4
  conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx]};
  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

void preprocess_forward_colmap(int P, int D, int M, const float* means3D, const glm::vec3* scales,
    const float scale_modifier, const glm::vec4* rotations, const float* opacities, const float* shs, bool* clamped,
    const float* cov3D_precomp, const float* colors_precomp, const float* viewmatrix, const float* projmatrix,
    const glm::vec3* cam_pos, const int W, int H, const float focal_x, float focal_y, const float tan_fovx,
    float tan_fovy, int* radii, float2* means2D, float* depths, float* cov3Ds, float* rgb, float4* conic_opacity,
    const dim3 grid, uint32_t* tiles_touched, bool prefiltered) {
  preprocessCUDA_colmap<NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(P, D, M, means3D, scales, scale_modifier,
      rotations, opacities, shs, clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix, cam_pos, W, H,
      tan_fovx, tan_fovy, focal_x, focal_y, radii, means2D, depths, cov3Ds, rgb, conic_opacity, grid, tiles_touched,
      prefiltered);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other backward steps contained in preprocess)
__global__ void computeCov2DCUDA_colmap(int P, const float3* means, const int* radii, const float* cov3Ds,
    const float h_x, float h_y, const float tan_fovx, float tan_fovy, const float* view_matrix, const float* dL_dconics,
    float3* dL_dmeans, float* dL_dcov) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !(radii[idx] > 0)) return;

  // Reading location of 3D covariance for this Gaussian
  const float* cov3D = cov3Ds + 6 * idx;

  // Fetch gradients, recompute 2D covariance and relevant intermediate forward results needed in the backward.
  float3 mean      = means[idx];
  float3 dL_dconic = {dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3]};
  float3 t         = transformPoint4x3_colmap(mean, view_matrix);

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x              = min(limx, max(-limx, txtz)) * t.z;
  t.y              = min(limy, max(-limy, tytz)) * t.z;

  const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
  const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

  glm::mat3 J =
      glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z), 0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z), 0, 0, 0);

  glm::mat3 W = glm::mat3(view_matrix[0], view_matrix[4], view_matrix[8], view_matrix[1], view_matrix[5],
      view_matrix[9], view_matrix[2], view_matrix[6], view_matrix[10]);

  glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 T = W * J;

  glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

  // Use helper variables for 2D covariance entries. More compact.
  float a = cov2D[0][0] += 0.3f;
  float b = cov2D[0][1];
  float c = cov2D[1][1] += 0.3f;

  float denom = a * c - b * b;
  float dL_da = 0, dL_db = 0, dL_dc = 0;
  float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

  if (denom2inv != 0) {
    // Gradients of loss w.r.t. entries of 2D covariance matrix,
    // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
    // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
    dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
    dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
    dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
    // given gradients w.r.t. 2D covariance matrix (diagonal).
    // cov2D = transpose(T) * transpose(Vrk) * T;
    dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
    dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
    dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
    // given gradients w.r.t. 2D covariance matrix (off-diagonal).
    // Off-diagonal elements appear twice --> double the gradient.
    // cov2D = transpose(T) * transpose(Vrk) * T;
    dL_dcov[6 * idx + 1] =
        2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
    dL_dcov[6 * idx + 2] =
        2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
    dL_dcov[6 * idx + 4] =
        2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
  } else {
    for (int i = 0; i < 6; i++) dL_dcov[6 * idx + i] = 0;
  }

  // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
  // cov2D = transpose(T) * transpose(Vrk) * T;
  float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
                  (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
  float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
                  (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
  float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
                  (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
  float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
                  (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
  float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
                  (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
  float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
                  (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

  // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
  // T = W * J
  float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
  float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
  float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
  float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

  float tz  = 1.f / t.z;
  float tz2 = tz * tz;
  float tz3 = tz2 * tz;

  // Gradients of loss w.r.t. transformed Gaussian mean t
  float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
  float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
  float dL_dtz =
      -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

  // Account for transformation of mean to t
  // t = transformPoint4x3(mean, view_matrix);
  float3 dL_dmean = transformVec4x3Transpose_colmap({dL_dtx, dL_dty, dL_dtz}, view_matrix);

  // Gradients of loss w.r.t. Gaussian means, but only the portion
  // that is caused because the mean affects the covariance matrix.
  // Additional mean gradient is accumulated in BACKWARD::preprocess.
  dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 3D covariance matrix for each Gaussian.
__device__ void computeCov3D_colmap(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot,
    const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots) {
  // Recompute (intermediate) results for the 3D covariance computation.
  glm::vec4 q = rot;  // / glm::length(rot);
  float x     = q.x;
  float y     = q.y;
  float z     = q.z;
  float r     = q.w;

  glm::mat3 R = glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
      2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x), 2.f * (x * z - r * y),
      2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

  glm::mat3 S = glm::mat3(1.0f);

  glm::vec3 s = mod * scale;
  S[0][0]     = s.x;
  S[1][1]     = s.y;
  S[2][2]     = s.z;

  glm::mat3 M = S * R;

  const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

  glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
  glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

  // Convert per-element covariance loss gradients to matrix form
  glm::mat3 dL_dSigma = glm::mat3(dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[1],
      dL_dcov3D[3], 0.5f * dL_dcov3D[4], 0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]);

  // Compute loss gradient w.r.t. matrix M
  // dSigma_dM = 2 * M
  glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

  glm::mat3 Rt     = glm::transpose(R);
  glm::mat3 dL_dMt = glm::transpose(dL_dM);

  // Gradients of loss w.r.t. scale
  glm::vec3* dL_dscale = dL_dscales + idx;
  dL_dscale->x         = glm::dot(Rt[0], dL_dMt[0]);
  dL_dscale->y         = glm::dot(Rt[1], dL_dMt[1]);
  dL_dscale->z         = glm::dot(Rt[2], dL_dMt[2]);

  dL_dMt[0] *= s.x;
  dL_dMt[1] *= s.y;
  dL_dMt[2] *= s.z;

  // Gradients of loss w.r.t. normalized quaternion
  glm::vec4 dL_dq;
  dL_dq.x = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) +
            2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
  dL_dq.y = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) +
            2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
  dL_dq.z = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) +
            2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
  dL_dq.w = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) +
            2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);

  // Gradients of loss w.r.t. unnormalized quaternion
  float4* dL_drot = (float4*) (dL_drots + idx);
  *dL_drot        = float4{dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w};
  // dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except for the covariance computation and inversion
// (those are handled by a previous kernel call)
template <int C>
__global__ void preprocessCUDA_backward_colmap(int P, int D, int M, const float3* means, const int* radii,
    const float* shs, const bool* clamped, const glm::vec3* scales, const glm::vec4* rotations,
    const float scale_modifier, const float* proj, const glm::vec3* campos, const float3* dL_dmean2D,
    glm::vec3* dL_dmeans, float* dL_dcolor, float* dL_dcov3D, float* dL_dsh, glm::vec3* dL_dscale, glm::vec4* dL_drot) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !(radii[idx] > 0)) return;

  float3 m = means[idx];

  // Taking care of gradients from the screenspace points
  float4 m_hom = transformPoint4x4_colmap(m, proj);
  float m_w    = 1.0f / (m_hom.w + 0.0000001f);

  // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
  // from rendering procedure
  glm::vec3 dL_dmean;
  float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
  float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
  dL_dmean.x =
      (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
  dL_dmean.y =
      (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
  dL_dmean.z =
      (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

  // That's the second part of the mean gradient.
  // Previous computation of cov2D and following SH conversion also affects it.
  dL_dmeans[idx] += dL_dmean;

  // Compute gradient updates due to computing colors from SHs
  if (shs)
    computeColorFromSH(idx, D, M, (glm::vec3*) means, *campos, shs, clamped, (glm::vec3*) dL_dcolor,
        (glm::vec3*) dL_dmeans, (glm::vec3*) dL_dsh);

  // Compute gradient updates due to computing covariance from scale/rotation
  if (scales) computeCov3D_colmap(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

void preprocess_backward_colmap(int P, int D, int M, const float3* means3D, const int* radii, const float* shs,
    const bool* clamped, const glm::vec3* scales, const glm::vec4* rotations, const float scale_modifier,
    const float* cov3Ds, const float* viewmatrix, const float* projmatrix, const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy, const glm::vec3* campos, const float3* dL_dmean2D, const float* dL_dconic,
    glm::vec3* dL_dmean3D, float* dL_dcolor, float* dL_dcov3D, float* dL_dsh, glm::vec3* dL_dscale,
    glm::vec4* dL_drot) {
  // Propagate gradients for the path of 2D conic matrix computation.
  // Somewhat long, thus it is its own kernel rather than being part of "preprocess".
  // When done, loss gradient w.r.t. 3D means has been modified and
  // gradient w.r.t. 3D covariance matrix has been computed.
  computeCov2DCUDA_colmap KERNEL_ARG((P + 255) / 256, 256)(P, means3D, radii, cov3Ds, focal_x, focal_y, tan_fovx,
      tan_fovy, viewmatrix, dL_dconic, (float3*) dL_dmean3D, dL_dcov3D);

  // Propagate gradients for remaining steps: finish 3D mean gradients,
  // propagate color gradients to SH (if desireD), propagate 3D covariance matrix gradients to scale and rotation.
  preprocessCUDA_backward_colmap<NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(P, D, M, (float3*) means3D, radii, shs,
      clamped, (glm::vec3*) scales, (glm::vec4*) rotations, scale_modifier, projmatrix, campos, (float3*) dL_dmean2D,
      (glm::vec3*) dL_dmean3D, dL_dcolor, dL_dcov3D, dL_dsh, dL_dscale, dL_drot);
}

}  // namespace GaussianRasterizer