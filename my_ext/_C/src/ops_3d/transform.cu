/* from nvdiffrec/render/renderutils/c_src
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

#include <cuda.h>
#include <stdio.h>

#include "util.cuh"
//------------------------------------------------------------------------
// Kernels

template <typename T, bool is_points = true, bool to_homo = false, int N = 3, int M = 4>
__global__ void xfm_forward_kernel(uint32_t batch, uint32_t num, uint32_t matrix_stride, const T* __restrict__ points,
    const T* __restrict__ matrix, T* __restrict__ outputs) {
  unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int b = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ T mtx[M * M];
  if (threadIdx.x < M * M) mtx[threadIdx.x] = matrix[b * matrix_stride + threadIdx.x];
  __syncthreads();
  if (b >= batch || n >= num) return;

  points += (b * num + n) * N;
  outputs += (b * num + n) * (N + to_homo);
  T pos[N];
#pragma unroll
  for (int i = 0; i < N; ++i) pos[i] = points[i];

#pragma unroll
  for (int i = 0; i < N + to_homo; ++i) {
    T sum = 0;
#pragma unroll
    for (int j = 0; j < N; ++j) sum += mtx[i * M + j] * pos[j];
    if constexpr (is_points && N != M) sum += mtx[i * M + N];
    outputs[i] = sum;
  }
}

template <typename T, bool is_points = true, bool to_homo = false, int N = 3, int M = 4>
__global__ void xfm_backward_kernel(uint32_t batch, uint32_t num, uint32_t matrix_stride, const T* __restrict__ points,
    const T* __restrict__ matrix, const T* __restrict__ grad_out, T* __restrict__ grad_points,
    T* __restrict__ grad_matrix) {
  int tid        = threadIdx.x;
  int lane_id    = tid % WARP_SIZE;
  unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int b = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ T mtx[M * M];
  if (tid < M * M) mtx[threadIdx.x] = matrix[b * matrix_stride + threadIdx.x];
  __syncthreads();
  if (b >= batch) return;

  points += (b * num + n) * N;
  grad_points += (b * num + n) * N;
  grad_out += (b * num + n) * (N + to_homo);

  T pos[N], g_out[N + to_homo];
#pragma unroll
  for (int i = 0; i < N + to_homo; ++i) g_out[i] = n < num ? grad_out[i] : T(0);

  if (grad_points != nullptr && n < num) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      T sum = 0;
#pragma unroll
      for (int j = 0; j < N + to_homo; ++j) sum += mtx[j * M + i] * g_out[j];
      grad_points[i] = sum;
    }
  }

  if (grad_matrix == nullptr) return;
  grad_matrix += b * matrix_stride;

#pragma unroll
  for (int i = 0; i < N; ++i) pos[i] = n < num ? points[i] : T(0);
  volatile __shared__ T smem[M * M][WARP_SIZE];
  T val;

#pragma unroll
  for (int i = 0; i < N + to_homo; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      val = pos[j] * g_out[i];
#pragma unroll
      for (int k = 16; k >= 1; k /= 2) val += shfl_xor<T>(val, k);
      if (lane_id == 0) smem[i * M + j][tid / WARP_SIZE] = val;
      if (tid < WARP_SIZE && tid >= (blockDim.x / WARP_SIZE)) smem[i * M + j][tid] = 0;
    }
    if (is_points && N != M) {
      val = g_out[i];
#pragma unroll
      for (int k = 16; k >= 1; k /= 2) val += shfl_xor<T>(val, k);
      if (lane_id == 0) smem[i * M + N][tid / WARP_SIZE] = val;
      if (tid < WARP_SIZE && tid >= (blockDim.x / WARP_SIZE)) smem[i * M + N][tid] = 0;
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N + to_homo; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      if (tid < WARP_SIZE) {
        val = smem[i * M + j][tid];
#pragma unroll
        for (int k = 16; k >= 1; k /= 2) val += shfl_xor<T>(val, k);
        if (tid == 0) atomicAdd(grad_matrix + i * M + j, val);
      }
    }
    if (is_points && N != M) {
      if (tid < WARP_SIZE) {
        val = smem[i * M + N][tid];
#pragma unroll
        for (int k = 16; k >= 1; k /= 2) val += shfl_xor<T>(val, k);
        if (tid == 0) atomicAdd(grad_matrix + i * M + N, val);
      }
    }
  }
}

Tensor xfm_fwd(const Tensor& points, const Tensor& matrix, bool isPoints, bool to_homo) {
  CHECK_CUDA(points);
  CHECK_CUDA(matrix);
  BCNN_ASSERT(points.dtype() == matrix.dtype(), "dtype for points and matrix must be same");
  BCNN_ASSERT(points.ndimension() >= 2, "Error shape for points");

  const uint32_t DP = points.size(-1), DM = matrix.size(-1);
  const uint32_t matrix_stride = matrix.numel() == DM * DM ? 0 : DM * DM;
  const uint32_t N             = matrix_stride == 0 ? points.numel() / DP : points.size(-2);
  const uint32_t B             = matrix_stride == 0 ? 1 : points.numel() / (N * DP);

  BCNN_ASSERT(matrix.ndimension() >= 2 && matrix.size(-2) == DM, "Error shape for matrix");
  BCNN_ASSERT(matrix.numel() == B * DM * DM || matrix.numel() == DM * DM, "Error shape for matrix");
  BCNN_ASSERT((DP == 3 || DP == 4) && (DM == 4), "Error shape for points or matrix");

  auto out_shape = points.sizes().vec();
  if (to_homo) out_shape[out_shape.size() - 1]++;
  Tensor out = torch::zeros(out_shape, points.options());

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "xfm_fwd", [&] {
    dim3 block(256, 1, 1);
    dim3 grid(div_round_up(N, 256u), B, 1);
    if (isPoints && to_homo && DP == 3 && DM == 4) {  // ([4, 4] @ [N,3].T).T = [N, 4]
      xfm_forward_kernel<scalar_t, true, true, 3, 4> KERNEL_ARG(grid, block)(B, N, matrix_stride,
          points.contiguous().data<scalar_t>(), matrix.contiguous().data<scalar_t>(), out.data<scalar_t>());
    } else if (isPoints && !to_homo && DP == 3 && DM == 4) {  // ([4, 4] @ [N,3].T).T = [N, 3]
      xfm_forward_kernel<scalar_t, true, false, 3, 4> KERNEL_ARG(grid, block)(B, N, matrix_stride,
          points.contiguous().data<scalar_t>(), matrix.contiguous().data<scalar_t>(), out.data<scalar_t>());
    } else if (isPoints && !to_homo && DP == 4 && DM == 4) {  // ([4, 4] @ [N,4].T).T = [N, 4]
      xfm_forward_kernel<scalar_t, true, false, 4, 4> KERNEL_ARG(grid, block)(B, N, matrix_stride,
          points.contiguous().data<scalar_t>(), matrix.contiguous().data<scalar_t>(), out.data<scalar_t>());
    } else if (!isPoints && !to_homo && DP == 3 && DM == 4) {  // ([4, 4] @ [N,3].T).T = [N, 3]
      xfm_forward_kernel<scalar_t, false, false, 3, 4> KERNEL_ARG(grid, block)(B, N, matrix_stride,
          points.contiguous().data<scalar_t>(), matrix.contiguous().data<scalar_t>(), out.data<scalar_t>());
    } else if (!isPoints && to_homo && DP == 3 && DM == 4) {  // ([4, 4] @ [N,3].T).T = [N, 3]
      xfm_forward_kernel<scalar_t, false, false, 3, 4> KERNEL_ARG(grid, block)(B, N, matrix_stride,
          points.contiguous().data<scalar_t>(), matrix.contiguous().data<scalar_t>(), out.data<scalar_t>());
    } else {
      BCNN_ASSERT(false, "Unsupport configure");
    }
    CHECK_CUDA_ERROR("xfm_forward_kernel");
  });
  return out;
}

vector<at::optional<Tensor>> xfm_bwd(const Tensor& grad_out, const Tensor& points, const Tensor& matrix, bool isPoints,
    bool to_homo, bool need_grad_points, bool need_grad_matrix) {
  at::optional<Tensor> grad_points, grad_matrix;
  if (need_grad_points) grad_points = torch::zeros_like(points);
  if (need_grad_matrix) grad_matrix = torch::zeros_like(matrix);

  const uint32_t DP = points.size(-1), DM = matrix.size(-1);
  const uint32_t matrix_stride = matrix.numel() == DM * DM ? 0 : DM * DM;
  const uint32_t N             = matrix_stride == 0 ? points.numel() / DP : points.size(-2);
  const uint32_t B             = matrix_stride == 0 ? 1 : points.numel() / (N * DP);

  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "xfm_bwd", [&] {
    dim3 block(256, 1, 1);
    dim3 grid(div_round_up(N, 256u), B, 1);
    if (isPoints && to_homo && DP == 3 && DM == 4) {  // ([4, 4] @ [N,3].T).T = [N, 4]
      xfm_backward_kernel<scalar_t, true, true, 3, 4> KERNEL_ARG(grid, block)(B, N, matrix_stride,
          points.contiguous().data<scalar_t>(), matrix.contiguous().data<scalar_t>(),
          grad_out.contiguous().data<scalar_t>(), need_grad_points ? grad_points.value().data<scalar_t>() : nullptr,
          need_grad_matrix ? grad_matrix.value().data<scalar_t>() : nullptr);
    } else if (isPoints && !to_homo && DP == 3 && DM == 4) {  // ([4, 4] @ [N,3].T).T = [N, 3]
      xfm_backward_kernel<scalar_t, true, false, 3, 4> KERNEL_ARG(grid, block)(B, N, matrix_stride,
          points.contiguous().data<scalar_t>(), matrix.contiguous().data<scalar_t>(),
          grad_out.contiguous().data<scalar_t>(), need_grad_points ? grad_points.value().data<scalar_t>() : nullptr,
          need_grad_matrix ? grad_matrix.value().data<scalar_t>() : nullptr);
    } else if (isPoints && !to_homo && DP == 4 && DM == 4) {  // ([4, 4] @ [N,4].T).T = [N, 4]
      xfm_backward_kernel<scalar_t, true, false, 4, 4> KERNEL_ARG(grid, block)(B, N, matrix_stride,
          points.contiguous().data<scalar_t>(), matrix.contiguous().data<scalar_t>(),
          grad_out.contiguous().data<scalar_t>(), need_grad_points ? grad_points.value().data<scalar_t>() : nullptr,
          need_grad_matrix ? grad_matrix.value().data<scalar_t>() : nullptr);
    } else if (!isPoints && !to_homo && DP == 3 && DM == 4) {  // ([4, 4] @ [N,3].T).T = [N, 3]
      xfm_backward_kernel<scalar_t, false, false, 3, 4> KERNEL_ARG(grid, block)(B, N, matrix_stride,
          points.contiguous().data<scalar_t>(), matrix.contiguous().data<scalar_t>(),
          grad_out.contiguous().data<scalar_t>(), need_grad_points ? grad_points.value().data<scalar_t>() : nullptr,
          need_grad_matrix ? grad_matrix.value().data<scalar_t>() : nullptr);
    } else if (!isPoints && to_homo && DP == 3 && DM == 4) {  // ([4, 4] @ [N,3].T).T = [N, 3]
      xfm_backward_kernel<scalar_t, false, false, 3, 4> KERNEL_ARG(grid, block)(B, N, matrix_stride,
          points.contiguous().data<scalar_t>(), matrix.contiguous().data<scalar_t>(),
          grad_out.contiguous().data<scalar_t>(), need_grad_points ? grad_points.value().data<scalar_t>() : nullptr,
          need_grad_matrix ? grad_matrix.value().data<scalar_t>() : nullptr);
    } else {
      BCNN_ASSERT(false, "Unsupport configure");
    }
    CHECK_CUDA_ERROR("xfm_backward_kernel");
  });
  return {grad_points, grad_matrix};
}

REGIST_PYTORCH_EXTENSION(ops_3d_transform, {
  m.def("xfm_fwd", &xfm_fwd, "xfm_fwd (CUDA)");
  m.def("xfm_bwd", &xfm_bwd, "xfm_bwd (CUDA)");
});