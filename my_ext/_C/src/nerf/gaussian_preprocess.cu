#if __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "ops_3d.h"
#include "util.cuh"
namespace cg = cooperative_groups;

namespace GaussianRasterizer {

__forceinline__ __device__ float ndc2Pix(float v, int S) { return ((v + 1.0) * S - 1.0) * 0.5; }

__forceinline__ __device__ bool in_frustum(int idx, const float* orig_points, const float* viewmatrix,
    const float* projmatrix, bool prefiltered, float3& p_view) {
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};

  // Bring points to screen space
  // float4 p_hom = xfm_p_4x4(p_orig, projmatrix);
  // float p_w    = 1.0f / (p_hom.w + 0.0000001f);
  // float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
  p_view = xfm_p_4x3(p_orig, viewmatrix);

  // if (p_view.z <= 0.2f)  // || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
  if (p_view.z <= -1.0f) {
    if (prefiltered) {
      printf("Point is filtered although prefiltered is set. This shouldn't happen!");
      __trap();
    }
    return false;
  }
  return true;
}

// Forward version of 2D covariance matrix computation
template <typename T = float, typename T3 = float3>
__device__ T3 computeCov2D(
    const T3& mean, T focal_x, T focal_y, T tan_fovx, T tan_fovy, const T* cov3D, const T* viewmatrix) {
  // The following models the steps outlined by equations 29 and 31 in "EWA Splatting" (Zwicker et al., 2002).
  // Additionally considers aspect / scaling of viewport. Transposes used to account for row-/column-major conventions.
  T3 t = xfm_p_4x3(mean, viewmatrix);

  const T limx = 1.3f * tan_fovx;
  const T limy = 1.3f * tan_fovy;
  const T txtz = t.x / t.z;
  const T tytz = t.y / t.z;
  t.x          = clamp(txtz, -limx, limx) * t.z;
  t.y          = clamp(tytz, -limy, limy) * t.z;

  T J[9] = {focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z), 0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z), 0,
      0, 0};

  T W[9] = {viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[8],
      viewmatrix[9], viewmatrix[10]};

  T M[9] = {0};
  matmul_3x3x3(W, J, M);

  T Vrk[9] = {cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]};

  T* tmp = J;
  zero_mat3(tmp);
  matmul_3x3x3_tn(M, Vrk, tmp);
  T* cov = Vrk;
  zero_mat3(cov);
  matmul_3x3x3(tmp, M, cov);

  // Apply low-pass filter: every Gaussian should be at least one pixel wide/high. Discard 3rd row and column.
  cov[0] += 0.3f;
  cov[4] += 0.3f;
  return {T(cov[0]), T(cov[1]), T(cov[4])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care of quaternion normalization.
template <typename T, typename T3, typename T4>
__device__ void computeCov3D(const T3& s, const T4& rot, T* cov3D) {
  // Create scaling matrix
  T R[9] = {0};
  quaternion_to_R(rot, R);  // Normalize quaternion to get valid rotation

  T sx2 = s.x * s.x;
  T sy2 = s.y * s.y;
  T sz2 = s.z * s.z;

  // Covariance is symmetric, only store upper right
  cov3D[0] = R[0] * R[0] * sx2 + R[1] * R[1] * sy2 + R[2] * R[2] * sz2;
  cov3D[1] = R[0] * R[3] * sx2 + R[1] * R[4] * sy2 + R[2] * R[5] * sz2;
  cov3D[2] = R[0] * R[6] * sx2 + R[1] * R[7] * sy2 + R[2] * R[8] * sz2;
  cov3D[3] = R[3] * R[3] * sx2 + R[4] * R[4] * sy2 + R[5] * R[5] * sz2;
  cov3D[4] = R[3] * R[6] * sx2 + R[4] * R[7] * sy2 + R[5] * R[8] * sz2;
  cov3D[5] = R[6] * R[6] * sx2 + R[7] * R[7] * sy2 + R[8] * R[8] * sz2;
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA(int P, int D, int M, const float* orig_points, const float3* scales,
    const float scale_modifier, const float4* rotations, const float* opacities, const float* shs, bool* clamped,
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
  if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) return;

  // Transform point by projecting
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
  float4 p_hom  = xfm_p_4x4<float, float3, float4>(p_orig, projmatrix);
  float p_w     = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

  // If 3D covariance matrix is precomputed, use it, otherwise compute from scaling and rotation parameters.
  const float* cov3D;
  if (cov3D_precomp != nullptr) {
    cov3D = cov3D_precomp + idx * 6;
  } else {
    computeCov3D<float, float3, float4>(scales[idx], rotations[idx], cov3Ds + idx * 6);
    cov3D = cov3Ds + idx * 6;
  }

  // Compute 2D screen-space covariance matrix
  float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

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

void preprocess_forward(int P, int D, int M, const float* means3D, const float3* scales, const float scale_modifier,
    const float4* rotations, const float* opacities, const float* shs, bool* clamped, const float* cov3D_precomp,
    const float* colors_precomp, const float* viewmatrix, const float* projmatrix, const glm::vec3* cam_pos,
    const int W, int H, const float focal_x, float focal_y, const float tan_fovx, float tan_fovy, int* radii,
    float2* means2D, float* depths, float* cov3Ds, float* rgb, float4* conic_opacity, const dim3 grid,
    uint32_t* tiles_touched, bool prefiltered) {
  preprocessCUDA<NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(P, D, M, means3D, scales, scale_modifier, rotations,
      opacities, shs, clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix, cam_pos, W, H, tan_fovx, tan_fovy,
      focal_x, focal_y, radii, means2D, depths, cov3Ds, rgb, conic_opacity, grid, tiles_touched, prefiltered);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P, const float3* means, const int* radii, const float* cov3Ds, const float fx,
    float fy, const float tan_fovx, float tan_fovy, const float* view_matrix, const float* dL_dconics,
    float3* dL_dmeans, float* dL_dcov) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !(radii[idx] > 0)) return;

  // Reading location of 3D covariance for this Gaussian
  const float* cov3D = cov3Ds + 6 * idx;

  // Fetch gradients, recompute 2D covariance and relevant intermediate forward results needed in the backward.
  float3 mean      = means[idx];
  float3 dL_dconic = {dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3]};
  float3 t         = xfm_p_4x3(mean, view_matrix);

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x              = clamp(txtz, -limx, limx) * t.z;
  t.y              = clamp(tytz, -limy, limy) * t.z;

  const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
  const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

  float J[9]   = {fx / t.z, 0.0f, -(fx * t.x) / (t.z * t.z), 0.0f, fy / t.z, -(fy * t.y) / (t.z * t.z), 0, 0, 0};
  float W[9]   = {view_matrix[0], view_matrix[1], view_matrix[2], view_matrix[4], view_matrix[5], view_matrix[6],
        view_matrix[8], view_matrix[9], view_matrix[10]};
  float Vrk[9] = {cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]};

  float T[9] = {0};
  matmul_3x3x3(W, J, T);
  float* tmp = J;
  zero_mat3(tmp);
  matmul_3x3x3_tn(T, Vrk, tmp);
  float cov2D[9] = {};
  matmul_3x3x3(tmp, T, cov2D);

  // Use helper variables for 2D covariance entries. More compact.
  float a = cov2D[0] += 0.3f;
  float b = cov2D[1];
  float c = cov2D[4] += 0.3f;

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
    dL_dcov[6 * idx + 0] = (T[0] * T[0] * dL_da + T[0] * T[1] * dL_db + T[1] * T[1] * dL_dc);
    dL_dcov[6 * idx + 3] = (T[1] * T[1] * dL_da + T[1] * T[4] * dL_db + T[4] * T[4] * dL_dc);
    dL_dcov[6 * idx + 5] = (T[6] * T[6] * dL_da + T[6] * T[7] * dL_db + T[7] * T[7] * dL_dc);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
    // given gradients w.r.t. 2D covariance matrix (off-diagonal).
    // Off-diagonal elements appear twice --> double the gradient.
    // cov2D = transpose(T) * transpose(Vrk) * T;
    dL_dcov[6 * idx + 1] = 2 * T[0] * T[3] * dL_da + (T[0] * T[4] + T[3] * T[1]) * dL_db + 2 * T[1] * T[4] * dL_dc;
    dL_dcov[6 * idx + 2] = 2 * T[0] * T[6] * dL_da + (T[0] * T[7] + T[6] * T[1]) * dL_db + 2 * T[1] * T[7] * dL_dc;
    dL_dcov[6 * idx + 4] = 2 * T[6] * T[3] * dL_da + (T[3] * T[7] + T[6] * T[4]) * dL_db + 2 * T[4] * T[7] * dL_dc;
  } else {
    for (int i = 0; i < 6; i++) dL_dcov[6 * idx + i] = 0;
  }

  // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
  // cov2D = transpose(T) * transpose(Vrk) * T;
  float dL_dT0 = 2 * (T[0] * cov3D[0] + T[3] * cov3D[1] + T[6] * cov3D[2]) * dL_da +
                 (T[1] * cov3D[0] + T[4] * cov3D[1] + T[7] * cov3D[2]) * dL_db;
  float dL_dT1 = 2 * (T[1] * cov3D[0] + T[4] * cov3D[1] + T[7] * cov3D[2]) * dL_dc +
                 (T[0] * cov3D[0] + T[3] * cov3D[1] + T[6] * cov3D[2]) * dL_db;
  float dL_dT2 = 0;
  float dL_dT3 = 2 * (T[0] * cov3D[1] + T[3] * cov3D[3] + T[6] * cov3D[4]) * dL_da +
                 (T[1] * cov3D[1] + T[4] * cov3D[3] + T[7] * cov3D[4]) * dL_db;
  float dL_dT4 = 2 * (T[1] * cov3D[1] + T[4] * cov3D[3] + T[7] * cov3D[4]) * dL_dc +
                 (T[0] * cov3D[3] + T[3] * cov3D[4] + T[6] * cov3D[5]) * dL_db;
  float dL_dT5 = 0;
  float dL_dT6 = 2 * (T[0] * cov3D[2] + T[3] * cov3D[4] + T[6] * cov3D[5]) * dL_da +
                 (T[1] * cov3D[2] + T[4] * cov3D[4] + T[7] * cov3D[5]) * dL_db;
  float dL_dT7 = 2 * (T[1] * cov3D[2] + T[4] * cov3D[4] + T[7] * cov3D[5]) * dL_dc +
                 (T[0] * cov3D[2] + T[3] * cov3D[4] + T[6] * cov3D[5]) * dL_db;
  float dL_dT8 = 0;

  // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
  // T = W * J
  float dL_dJ00 = W[0] * dL_dT0 + W[3] * dL_dT3 + W[6] * dL_dT6;
  float dL_dJ02 = W[0] * dL_dT2 + W[3] * dL_dT5 + W[6] * dL_dT8;
  float dL_dJ11 = W[1] * dL_dT1 + W[4] * dL_dT4 + W[7] * dL_dT7;
  float dL_dJ12 = W[1] * dL_dT2 + W[4] * dL_dT5 + W[7] * dL_dT8;

  float tz  = 1.f / t.z;
  float tz2 = tz * tz;
  float tz3 = tz2 * tz;

  // Gradients of loss w.r.t. transformed Gaussian mean t
  float dL_dtx = x_grad_mul * -fx * tz2 * dL_dJ02;
  float dL_dty = y_grad_mul * -fy * tz2 * dL_dJ12;
  float dL_dtz =
      -fx * tz2 * dL_dJ00 - fy * tz2 * dL_dJ11 + (2 * fx * t.x) * tz3 * dL_dJ02 + (2 * fy * t.y) * tz3 * dL_dJ12;

  // Account for transformation of mean to t
  // t = xfm_p_4x3(mean, view_matrix);
  float3 dL_dmean = xfm_v_4x3_T<float, float3>({dL_dtx, dL_dty, dL_dtz}, view_matrix);

  // Gradients of loss w.r.t. Gaussian means, but only the portion
  // that is caused because the mean affects the covariance matrix.
  // Additional mean gradient is accumulated in BACKWARD::preprocess.
  dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 3D covariance matrix for each Gaussian.
template <typename T, typename T3, typename T4>
__device__ void computeCov3D_backward(
    int idx, const T3 scale, const T4 rot, const T* dL_dcov3Ds, T3* dL_dscales, T4* dL_drots) {
  T R[9] = {0};
  quaternion_to_R(rot, R);  // Recompute (intermediate) results for the 3D covariance computation.

  const T* dL_dcov3D = dL_dcov3Ds + 6 * idx;
  T3 gs;
  gs.x = R[0] * R[0] * dL_dcov3D[0] + R[0] * R[3] * dL_dcov3D[1] + R[0] * R[6] * dL_dcov3D[2] +
         R[3] * R[3] * dL_dcov3D[3] + R[3] * R[6] * dL_dcov3D[4] + R[6] * R[6] * dL_dcov3D[5];
  gs.y = R[1] * R[1] * dL_dcov3D[0] + R[1] * R[4] * dL_dcov3D[1] + R[1] * R[7] * dL_dcov3D[2] +
         R[4] * R[4] * dL_dcov3D[3] + R[4] * R[7] * dL_dcov3D[4] + R[7] * R[7] * dL_dcov3D[5];
  gs.z = R[2] * R[2] * dL_dcov3D[0] + R[2] * R[5] * dL_dcov3D[1] + R[2] * R[8] * dL_dcov3D[2] +
         R[5] * R[5] * dL_dcov3D[3] + R[5] * R[8] * dL_dcov3D[4] + R[8] * R[8] * dL_dcov3D[5];
  gs.x *= 2 * scale.x;
  gs.y *= 2 * scale.y;
  gs.z *= 2 * scale.z;
  dL_dscales[idx] = gs;

  T sx2 = scale.x * scale.x;
  T sy2 = scale.y * scale.y;
  T sz2 = scale.z * scale.z;

  T dL_dR[9];
  dL_dR[0] = (2 * R[0] * dL_dcov3D[0] + R[3] * dL_dcov3D[1] + R[6] * dL_dcov3D[2]) * sx2;
  dL_dR[1] = (2 * R[1] * dL_dcov3D[0] + R[4] * dL_dcov3D[1] + R[7] * dL_dcov3D[2]) * sy2;
  dL_dR[2] = (2 * R[2] * dL_dcov3D[0] + R[5] * dL_dcov3D[1] + R[8] * dL_dcov3D[2]) * sz2;
  dL_dR[3] = (2 * R[3] * dL_dcov3D[3] + R[0] * dL_dcov3D[1] + R[6] * dL_dcov3D[4]) * sx2;
  dL_dR[4] = (2 * R[4] * dL_dcov3D[3] + R[1] * dL_dcov3D[1] + R[7] * dL_dcov3D[4]) * sy2;
  dL_dR[5] = (2 * R[5] * dL_dcov3D[3] + R[2] * dL_dcov3D[1] + R[8] * dL_dcov3D[4]) * sz2;
  dL_dR[6] = (2 * R[6] * dL_dcov3D[5] + R[0] * dL_dcov3D[2] + R[3] * dL_dcov3D[4]) * sx2;
  dL_dR[7] = (2 * R[7] * dL_dcov3D[5] + R[1] * dL_dcov3D[2] + R[4] * dL_dcov3D[4]) * sy2;
  dL_dR[8] = (2 * R[8] * dL_dcov3D[5] + R[2] * dL_dcov3D[2] + R[5] * dL_dcov3D[4]) * sz2;

  dL_drots[idx] = dL_quaternion_to_R(rot, dL_dR);  // Gradients of loss w.r.t. normalized quaternion
}

// Backward pass of the preprocessing steps, except for the covariance computation and inversion
// (those are handled by a previous kernel call)
template <int C>
__global__ void preprocessCUDA_backward(int P, int D, int M, const float3* means, const int* radii, const float* shs,
    const bool* clamped, const float3* scales, const float4* rotations, const float scale_modifier, const float* proj,
    const glm::vec3* campos, const float3* dL_dmean2D, glm::vec3* dL_dmeans, float* dL_dcolor, float* dL_dcov3D,
    float* dL_dsh, float3* dL_dscale, float4* dL_drot) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !(radii[idx] > 0)) return;

  float3 m = means[idx];

  // Taking care of gradients from the screenspace points
  float4 m_hom = xfm_p_4x4<float, float3, float4>(m, proj);
  float m_w    = 1.0f / (m_hom.w + 0.0000001f);

  // Compute loss gradient w.r.t. 3D means due to gradients of 2D means from rendering procedure
  glm::vec3 dL_dmean;
  float mul1 = (proj[0] * m.x + proj[1] * m.y + proj[2] * m.z + proj[3]) * m_w * m_w;
  float mul2 = (proj[4] * m.x + proj[5] * m.y + proj[6] * m.z + proj[7]) * m_w * m_w;
  dL_dmean.x =
      (proj[0] * m_w - proj[12] * mul1) * dL_dmean2D[idx].x + (proj[4] * m_w - proj[12] * mul2) * dL_dmean2D[idx].y;
  dL_dmean.y =
      (proj[1] * m_w - proj[13] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[13] * mul2) * dL_dmean2D[idx].y;
  dL_dmean.z =
      (proj[2] * m_w - proj[14] * mul1) * dL_dmean2D[idx].x + (proj[6] * m_w - proj[14] * mul2) * dL_dmean2D[idx].y;

  // That's the second part of the mean gradient.
  // Previous computation of cov2D and following SH conversion also affects it.
  dL_dmeans[idx] += dL_dmean;

  // Compute gradient updates due to computing colors from SHs
  if (shs)
    computeColorFromSH(idx, D, M, (glm::vec3*) means, *campos, shs, clamped, (glm::vec3*) dL_dcolor,
        (glm::vec3*) dL_dmeans, (glm::vec3*) dL_dsh);

  // Compute gradient updates due to computing covariance from scale/rotation
  if (scales)
    computeCov3D_backward<float, float3, float4>(idx, scales[idx], rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

void preprocess_backward(int P, int D, int M, const float3* means3D, const int* radii, const float* shs,
    const bool* clamped, const float3* scales, const float4* rotations, const float scale_modifier, const float* cov3Ds,
    const float* viewmatrix, const float* projmatrix, const float focal_x, float focal_y, const float tan_fovx,
    float tan_fovy, const glm::vec3* campos, const float3* dL_dmean2D, const float* dL_dconic, glm::vec3* dL_dmean3D,
    float* dL_dcolor, float* dL_dcov3D, float* dL_dsh, float3* dL_dscale, float4* dL_drot) {
  // Propagate gradients for the path of 2D conic matrix computation.
  // Somewhat long, thus it is its own kernel rather than being part of "preprocess".
  // When done, loss gradient w.r.t. 3D means has been modified and
  // gradient w.r.t. 3D covariance matrix has been computed.
  computeCov2DCUDA KERNEL_ARG((P + 255) / 256, 256)(P, means3D, radii, cov3Ds, focal_x, focal_y, tan_fovx, tan_fovy,
      viewmatrix, dL_dconic, (float3*) dL_dmean3D, dL_dcov3D);
  // cudaDeviceSynchronize();
  // printf("after computeCov2DCUDA");
  // Propagate gradients for remaining steps: finish 3D mean gradients,
  // propagate color gradients to SH (if desireD), propagate 3D covariance matrix gradients to scale and rotation.
  preprocessCUDA_backward<NUM_CHANNELS> KERNEL_ARG((P + 255) / 256, 256)(P, D, M, (float3*) means3D, radii, shs,
      clamped, (float3*) scales, (float4*) rotations, scale_modifier, projmatrix, campos, (float3*) dL_dmean2D,
      (glm::vec3*) dL_dmean3D, dL_dcolor, dL_dcov3D, dL_dsh, dL_dscale, dL_drot);

  // cudaDeviceSynchronize();
  // printf("after preprocessCUDA_backward");
}

template <typename T, typename T3, typename T4>
__global__ void compute_cov3D_forward_kernel(
    int N, const T* __restrict__ scale, const T* __restrict__ quaternion, T* __restrict__ cov3D) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    computeCov3D<T, T3, T4>(*((T3*) (scale + idx * 3)), *((T4*) (quaternion + idx * 4)), cov3D + idx * 6);
  }
}
Tensor compute_cov3D_forward(Tensor q, Tensor s) {
  CHECK_CUDA(q);
  CHECK_CUDA(s);
  auto shape              = s.sizes().vec();
  int N                   = s.numel() / 3;
  shape[shape.size() - 1] = 4;
  BCNN_ASSERT(q.sizes().vec() == shape && s.size(-1) == 3, "q.shape[:-1] == s.shape[:-1]");
  shape[shape.size() - 1] = 6;
  Tensor cov3D            = q.new_zeros(shape);

  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "compute_cov3D_forward", [&] {
    using scalar_t3 = TypeSelecotr<scalar_t>::T3;
    using scalar_t4 = TypeSelecotr<scalar_t>::T4;
    compute_cov3D_forward_kernel<scalar_t, scalar_t3, scalar_t4> KERNEL_ARG(div_round_up(N, 256), 256)(
        N, s.contiguous().data_ptr<scalar_t>(), q.contiguous().data_ptr<scalar_t>(), cov3D.data<scalar_t>());
  });
  return cov3D;
}

template <typename T, typename T3, typename T4>
__global__ void compute_cov3D_backward_kernel(int N, const T* __restrict__ quaternion, const T* __restrict__ scale,
    const T* __restrict__ dL_dcov3D, T* __restrict__ dL_dq, T* __restrict__ dL_ds) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    computeCov3D_backward<T, T3, T4>(
        idx, *((T3*) (scale + idx * 3)), *((T4*) (quaternion + idx * 4)), dL_dcov3D, (T3*) dL_ds, (T4*) dL_dq);
  }
}

vector<Tensor> compute_cov3D_backward(Tensor q, Tensor s, Tensor dL_dCov3D) {
  Tensor dq = torch::zeros_like(q);
  Tensor ds = torch::zeros_like(s);
  int N     = s.numel() / 3;
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "compute_cov3D_backward_kernel", [&] {
    using scalar_t3 = TypeSelecotr<scalar_t>::T3;
    using scalar_t4 = TypeSelecotr<scalar_t>::T4;
    compute_cov3D_backward_kernel<scalar_t, scalar_t3, scalar_t4> KERNEL_ARG(div_round_up(N, 256), 256)(N,
        q.contiguous().data_ptr<scalar_t>(), s.contiguous().data_ptr<scalar_t>(),
        dL_dCov3D.contiguous().data_ptr<scalar_t>(), dq.data<scalar_t>(), ds.data<scalar_t>());
  });
  return {dq, ds};
}

template <typename T = float, typename T3>
__global__ void compute_cov2D_forward_kernel(int N, const T* __restrict__ cov3D, const T* __restrict__ mean,
    const T* __restrict__ viewmatrix, T focal_x, T focal_y, T tan_fovx, T tan_fovy, T* __restrict__ cov2D) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    T3 t               = computeCov2D(*(T3*) (mean + idx * 3), focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
    cov2D[idx * 3 + 0] = t.x;
    cov2D[idx * 3 + 1] = t.y;
    cov2D[idx * 3 + 2] = t.z;
  }
}
Tensor compute_cov2D_forward(
    Tensor cov3D, Tensor mean, Tensor viewmatrix, float focal_x, float focal_y, float tan_fovx, float tan_fovy) {
  int N        = mean.numel() / 3;
  Tensor cov2D = torch::zeros_like(mean);

  AT_DISPATCH_FLOATING_TYPES(cov3D.scalar_type(), "compute_cov2D_forward", [&] {
    using scalar_t3 = TypeSelecotr<scalar_t>::T3;
    compute_cov2D_forward_kernel<scalar_t, scalar_t3> KERNEL_ARG(div_round_up(N, 256), 256)(N,
        cov3D.contiguous().data_ptr<scalar_t>(), mean.contiguous().data_ptr<scalar_t>(),
        viewmatrix.contiguous().data_ptr<scalar_t>(), focal_x, focal_y, tan_fovx, tan_fovy, cov2D.data<scalar_t>());
  });
  return cov2D;
}

template <typename T, typename T3>
__global__ void compute_cov2D_backward_kernel(int P, const T3* means, const T* cov3Ds, const T fx, T fy,
    const T tan_fovx, T tan_fovy, const T* view_matrix, const T* dL_dcov2D, T3* dL_dmeans, T* dL_dcov, T* dL_dvm) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Reading location of 3D covariance for this Gaussian
  const T* cov3D = cov3Ds + 6 * idx;

  // Fetch gradients, recompute 2D covariance and relevant intermediate forward results needed in the backward.
  T3 mean = means[idx];
  // T3 dL_dconic = {dL_dcov3D[4 * idx], dL_dcov3D[4 * idx + 1], dL_dconics[4 * idx + 3]};
  T3 t = xfm_p_4x3(mean, view_matrix);

  const T limx = 1.3f * tan_fovx;
  const T limy = 1.3f * tan_fovy;
  const T txtz = t.x / t.z;
  const T tytz = t.y / t.z;
  t.x          = clamp(txtz, -limx, limx) * t.z;
  t.y          = clamp(tytz, -limy, limy) * t.z;

  const T x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
  const T y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

  T J[9]   = {fx / t.z, 0.0f, -(fx * t.x) / (t.z * t.z), 0.0f, fy / t.z, -(fy * t.y) / (t.z * t.z), 0, 0, 0};
  T W[9]   = {view_matrix[0], view_matrix[1], view_matrix[2], view_matrix[4], view_matrix[5], view_matrix[6],
        view_matrix[8], view_matrix[9], view_matrix[10]};
  T Vrk[9] = {cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]};

  T M[9] = {0};
  matmul_3x3x3(W, J, M);
  T* tmp = J;
  zero_mat3(tmp);
  matmul_3x3x3_tn(M, Vrk, tmp);
  T cov2D[9] = {};  // = (JW) Vrk (JW)^T (WJ)^T Vrk WJ
  matmul_3x3x3(tmp, M, cov2D);

  // Use helper variables for 2D covariance entries. More compact.
  T a = cov2D[0] += 0.3f;
  T b = cov2D[1];
  T c = cov2D[4] += 0.3f;

  T denom = a * c - b * b;
  T dL_da = 0, dL_db = 0, dL_dc = 0;
  T denom2inv = 1.0f;  // / ((denom * denom) + 0.0000001f);

  if (denom2inv != 0) {
    // Gradients of loss w.r.t. entries of 2D covariance matrix,
    // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
    // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
    // dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
    // dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
    // dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);
    dL_da = dL_dcov2D[idx * 3 + 0];
    dL_db = dL_dcov2D[idx * 3 + 1];
    dL_dc = dL_dcov2D[idx * 3 + 2];

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
    // given gradients w.r.t. 2D covariance matrix (diagonal).
    // cov2D = transpose(M) * transpose(Vrk) * M;
    dL_dcov[6 * idx + 0] = (M[0] * M[0] * dL_da + M[0] * M[1] * dL_db + M[1] * M[1] * dL_dc);
    dL_dcov[6 * idx + 3] = (M[3] * M[3] * dL_da + M[3] * M[4] * dL_db + M[4] * M[4] * dL_dc);
    dL_dcov[6 * idx + 5] = (M[6] * M[6] * dL_da + M[6] * M[7] * dL_db + M[7] * M[7] * dL_dc);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
    // given gradients w.r.t. 2D covariance matrix (off-diagonal).
    // Off-diagonal elements appear twice --> double the gradient.
    // cov2D = transpose(M) * transpose(Vrk) * M;
    dL_dcov[6 * idx + 1] = 2 * M[0] * M[3] * dL_da + (M[0] * M[4] + M[3] * M[1]) * dL_db + 2 * M[1] * M[4] * dL_dc;
    dL_dcov[6 * idx + 2] = 2 * M[0] * M[6] * dL_da + (M[0] * M[7] + M[6] * M[1]) * dL_db + 2 * M[1] * M[7] * dL_dc;
    dL_dcov[6 * idx + 4] = 2 * M[6] * M[3] * dL_da + (M[3] * M[7] + M[6] * M[4]) * dL_db + 2 * M[4] * M[7] * dL_dc;
  } else {
    for (int i = 0; i < 6; i++) dL_dcov[6 * idx + i] = 0;
  }

  // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix M
  // cov2D = transpose(M) * transpose(Vrk) * M;
  T dL_dT0 = 2 * (M[0] * cov3D[0] + M[3] * cov3D[1] + M[6] * cov3D[2]) * dL_da +
             (M[1] * cov3D[0] + M[4] * cov3D[1] + M[7] * cov3D[2]) * dL_db;
  T dL_dT1 = 2 * (M[1] * cov3D[0] + M[4] * cov3D[1] + M[7] * cov3D[2]) * dL_dc +
             (M[0] * cov3D[0] + M[3] * cov3D[1] + M[6] * cov3D[2]) * dL_db;
  T dL_dT2 = 0;
  T dL_dT3 = 2 * (M[0] * cov3D[1] + M[3] * cov3D[3] + M[6] * cov3D[4]) * dL_da +
             (M[1] * cov3D[1] + M[4] * cov3D[3] + M[7] * cov3D[4]) * dL_db;
  T dL_dT4 = 2 * (M[1] * cov3D[1] + M[4] * cov3D[3] + M[7] * cov3D[4]) * dL_dc +
             (M[0] * cov3D[1] + M[3] * cov3D[3] + M[6] * cov3D[4]) * dL_db;
  T dL_dT5 = 0;
  T dL_dT6 = 2 * (M[0] * cov3D[2] + M[3] * cov3D[4] + M[6] * cov3D[5]) * dL_da +
             (M[1] * cov3D[2] + M[4] * cov3D[4] + M[7] * cov3D[5]) * dL_db;
  T dL_dT7 = 2 * (M[1] * cov3D[2] + M[4] * cov3D[4] + M[7] * cov3D[5]) * dL_dc +
             (M[0] * cov3D[2] + M[3] * cov3D[4] + M[6] * cov3D[5]) * dL_db;
  T dL_dT8 = 0;

  // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
  // M = W * J
  T dL_dJ00 = W[0] * dL_dT0 + W[3] * dL_dT3 + W[6] * dL_dT6;
  T dL_dJ02 = W[0] * dL_dT2 + W[3] * dL_dT5 + W[6] * dL_dT8;
  T dL_dJ11 = W[1] * dL_dT1 + W[4] * dL_dT4 + W[7] * dL_dT7;
  T dL_dJ12 = W[1] * dL_dT2 + W[4] * dL_dT5 + W[7] * dL_dT8;

  T tz  = 1.f / t.z;
  T tz2 = tz * tz;
  T tz3 = tz2 * tz;

  // Gradients of loss w.r.t. transformed Gaussian mean t
  T dL_dtx = x_grad_mul * -fx * tz2 * dL_dJ02;
  T dL_dty = y_grad_mul * -fy * tz2 * dL_dJ12;
  T dL_dtz = -fx * tz2 * dL_dJ00 - fy * tz2 * dL_dJ11 + (2 * fx * t.x) * tz3 * dL_dJ02 + (2 * fy * t.y) * tz3 * dL_dJ12;

  // Account for transformation of mean to t
  // t = xfm_p_4x3(mean, view_matrix);
  T3 dL_dmean = xfm_v_4x3_T<T, T3>({dL_dtx, dL_dty, dL_dtz}, view_matrix);

  // Gradients of loss w.r.t. Gaussian means, but only the portion
  // that is caused because the mean affects the covariance matrix.
  // Additional mean gradient is accumulated in BACKWARD::preprocess.
  dL_dmeans[idx] = dL_dmean;
}

vector<Tensor> compute_cov2D_backward(Tensor cov3D, Tensor mean, Tensor viewmatrix, float focal_x, float focal_y,
    float tan_fovx, float tan_fovy, Tensor grad_cov2D) {
  int N             = mean.numel() / 3;
  Tensor grad_cov3D = torch::zeros_like(cov3D);
  Tensor grad_mean  = torch::zeros_like(mean);
  Tensor grad_vm    = torch::zeros_like(viewmatrix);

  AT_DISPATCH_FLOATING_TYPES(cov3D.scalar_type(), "compute_cov2D_backward", [&] {
    using scalar_t3 = TypeSelecotr<scalar_t>::T3;
    compute_cov2D_backward_kernel<scalar_t, scalar_t3> KERNEL_ARG(div_round_up(N, 256), 256)(N,
        (scalar_t3*) mean.data_ptr<scalar_t>(), cov3D.data_ptr<scalar_t>(), focal_x, focal_y, tan_fovx, tan_fovy,
        viewmatrix.data_ptr<scalar_t>(), grad_cov2D.data_ptr<scalar_t>(), (scalar_t3*) grad_mean.data_ptr<scalar_t>(),
        grad_cov3D.data_ptr<scalar_t>(), grad_vm.data_ptr<scalar_t>());
  });

  return {grad_cov3D, grad_mean, grad_vm};
}
REGIST_PYTORCH_EXTENSION(nerf_gaussian_preprocess, {
  m.def("gs_compute_cov3D_forward", &compute_cov3D_forward, "compute_cov3D_forward (CUDA)");
  m.def("gs_compute_cov3D_backward", &compute_cov3D_backward, "compute_cov3D_backward (CUDA)");
  m.def("gs_compute_cov2D_forward", &compute_cov2D_forward, "compute_cov2D_forward (CUDA)");
  m.def("compute_cov2D_backward", &compute_cov2D_backward, "compute_cov2D_backward (CUDA)");
})
}  // namespace GaussianRasterizer