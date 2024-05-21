#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
template <typename scalar_t>
struct TypeSelecotr {
  using T  = float;
  using T3 = float3;
  using T4 = float4;
};

template <>
struct TypeSelecotr<float> {
  using T  = float;
  using T3 = float3;
  using T4 = float4;
};

template <>
struct TypeSelecotr<double> {
  using T  = double;
  using T3 = double3;
  using T4 = double4;
};
template <typename T>
__forceinline__ __device__ void zero_mat3(T* M) {
#pragma unroll
  for (int i = 0; i < 9; ++i) M[i] = 0;
}

template <typename T>
__forceinline__ __device__ void eye_mat3(T* M) {
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) M[i * 3 + j] = i == j;
}

template <typename T>
__forceinline__ __device__ void matmul_3x3x3(T* A, T* B, T* C) {
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
      }
    }
  }
}

template <typename T>
__forceinline__ __device__ void matmul_3x3x3_tn(T* At, T* B, T* C) {
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        C[i * 3 + j] += At[k * 3 + i] * B[k * 3 + j];
      }
    }
  }
}

template <typename T>
__forceinline__ __device__ void matmul_4x4x4(T* A, T* B, T* C) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        C[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
      }
    }
  }
}
template <typename T = float, typename T4 = float4>
__forceinline__ __device__ T4 quaternion_normalize(const T4& q) {
  T norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  norm   = T(1.) / max(sqrt(norm), T(1e-12));
  return T4(q.x * norm, q.y * norm, q.z * norm, q.w * norm);
}

template <typename T = float, typename T4 = float4>
__forceinline__ __device__ T4 quaternion_normalize(const T* q) {
  T norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  norm   = T(1.) / max(sqrt(norm), T(1e-12));
  return T4(q[0] * norm, q[1] * norm, q[2] * norm, q[3] * norm);
}

template <typename T = float, typename T4 = float4>
__forceinline__ __device__ void quaternion_to_R(const T4& q, T* R) {
  R[0] = 1 - 2 * (q.y * q.y + q.z * q.z);
  R[1] = 2 * (q.x * q.y - q.z * q.w);
  R[2] = 2 * (q.y * q.w + q.x * q.z);
  R[3] = 2 * (q.x * q.y + q.z * q.w);
  R[4] = 1 - 2 * (q.x * q.x + q.z * q.z);
  R[5] = 2 * (q.y * q.z - q.x * q.w);
  R[6] = 2 * (q.x * q.z - q.y * q.w);
  R[7] = 2 * (q.x * q.w + q.y * q.z);
  R[8] = 1 - 2 * (q.x * q.x + q.y * q.y);
}

template <typename T = float>
__forceinline__ __device__ void quaternion_to_R(const T* q, T* R) {
  R[0] = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
  R[1] = 2 * (q[0] * q[1] - q[2] * q[3]);
  R[2] = 2 * (q[1] * q[3] + q[0] * q[2]);
  R[3] = 2 * (q[0] * q[1] + q[2] * q[3]);
  R[4] = 1 - 2 * (q[0] * q[0] + q[2] * q[2]);
  R[5] = 2 * (q[1] * q[2] - q[0] * q[3]);
  R[6] = 2 * (q[0] * q[2] - q[1] * q[3]);
  R[7] = 2 * (q[0] * q[3] + q[1] * q[2]);
  R[8] = 1 - 2 * (q[0] * q[0] + q[1] * q[1]);
}

template <typename T = float, typename T4 = float4>
__forceinline__ __device__ T4 dL_quaternion_to_R(const T4& q, const T* dR) {
  T4 dq;
  dq.x = 2 * (-2 * q.x * (dR[4] + dR[8]) + q.y * (dR[1] + dR[3]) + q.z * (dR[2] + dR[6]) + q.w * (dR[7] - dR[5]));
  dq.y = 2 * (q.x * (dR[1] + dR[3]) - 2 * q.y * (dR[0] + dR[8]) + q.z * (dR[5] + dR[7]) + q.w * (dR[2] - dR[6]));
  dq.z = 2 * (q.x * (dR[2] + dR[6]) + q.y * (dR[5] + dR[7]) - 2 * q.z * (dR[0] + dR[4]) + q.w * (dR[3] - dR[1]));
  dq.w = 2 * (q.x * (dR[7] - dR[5]) + q.y * (dR[2] - dR[6]) + q.z * (dR[3] - dR[1]));
  return dq;
}

template <typename T = float>
__forceinline__ __host__ __device__ void dL_quaternion_to_R(const T* q, const T* dR, T* dq) {
  dq[0] = 2 * (-2 * q[0] * (dR[4] + dR[8]) + q[1] * (dR[1] + dR[3]) + q[2] * (dR[2] + dR[6]) + q[3] * (dR[7] - dR[5]));
  dq[1] = 2 * (q[0] * (dR[1] + dR[3]) - 2 * q[1] * (dR[0] + dR[8]) + q[2] * (dR[5] + dR[7]) + q[3] * (dR[2] - dR[6]));
  dq[2] = 2 * (q[0] * (dR[2] + dR[6]) + q[1] * (dR[5] + dR[7]) - 2 * q[2] * (dR[0] + dR[4]) + q[3] * (dR[3] - dR[1]));
  dq[3] = 2 * (q[0] * (dR[7] - dR[5]) + q[1] * (dR[2] - dR[6]) + q[2] * (dR[3] - dR[1]));
}

template <typename T, typename T3>
__forceinline__ __device__ T3 xfm_p_4x3(const T3& p, const T* matrix) {
  T3 transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11],
  };
  return transformed;
}

template <typename T, typename T3, typename T4>
__forceinline__ __device__ T4 xfm_p_4x4(const T3& p, const T* matrix) {
  T4 transformed = {matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11],
      matrix[12] * p.x + matrix[13] * p.y + matrix[14] * p.z + matrix[15]};
  return transformed;
}

template <typename T, typename T3>
__forceinline__ __device__ T3 xfm_v_4x3(const T3& p, const T* matrix) {
  T3 transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

template <typename T, typename T3>
__forceinline__ __device__ T3 xfm_v_4x3_T(const T3& p, const T* matrix) {
  T3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
  };
  return transformed;
}
