#include "ops_3d.h"
#include "util.cuh"
#include "glm_helper.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
namespace OPS_3D {

template <typename T>
__global__ void quaternion_to_R_forward_kernel(int N, const T *__restrict__ q, T *__restrict__ R) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    q      = q + idx * 4;
    R      = R + idx * 9;
    T norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    norm   = T(1.) / max(sqrt(norm), T(1e-12));
    T x = q[0] * norm, y = q[1] * norm, z = q[2] * norm, w = q[3] * norm;

    R[0] = 1 - 2 * (y * y + z * z);
    R[1] = 2 * (x * y - z * w);
    R[2] = 2 * (y * w + x * z);
    R[3] = 2 * (x * y + z * w);
    R[4] = 1 - 2 * (x * x + z * z);
    R[5] = 2 * (y * z - x * w);
    R[6] = 2 * (x * z - y * w);
    R[7] = 2 * (x * w + y * z);
    R[8] = 1 - 2 * (x * x + y * y);
  }
}

Tensor quaternion_to_R_forward(const Tensor &q) {
  // CHECK_CUDA(q);
  BCNN_ASSERT(q.size(-1) == 4, "Error shape for q");
  int N      = q.numel() / 4;
  auto shape = q.sizes().vec();
  shape.pop_back();
  shape.push_back(3);
  shape.push_back(3);
  Tensor R = q.new_zeros(shape);

  if (N == 0) return R;
  if (q.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "quaternion_to_R_forward", [&] {
      quaternion_to_R_forward_kernel KERNEL_ARG(div_round_up(N, 256), 256)(
          N, q.contiguous().data_ptr<scalar_t>(), R.data_ptr<scalar_t>());
      CHECK_CUDA_ERROR("quaternion_to_R_forward_kernel");
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "quaternion_to_R_forward", [&] {
      auto q_ptr = q.contiguous().data_ptr<scalar_t>();
      auto o_ptr = R.data_ptr<scalar_t>();
      for (int i = 0; i < N; ++i) {
        glm::qua qi(q_ptr[i * 4 + 0], q_ptr[i * 4 + 1], q_ptr[i * 4 + 2], q_ptr[i * 4 + 3]);
        GLMTypes<scalar_t>::mat3 matrix = glm::mat3_cast(glm::normalize(qi));  // NOTE: xyzw format
        mat3_copy_to(matrix, o_ptr + i * 9);
      }
    });
  }
  return R;
}

template <typename T>
__forceinline__ __host__ __device__ void quaternion_to_R_backward_function(
    const T *__restrict__ q, const T *__restrict__ dR, T *__restrict__ dq) {
  T x = q[0], y = q[1], z = q[2], w = q[3];

  T dx = (-T(2) * x * (dR[4] + dR[8]) + y * (dR[1] + dR[3]) + z * (dR[2] + dR[6]) + w * (dR[7] - dR[5]));
  T dy = (x * (dR[1] + dR[3]) - T(2) * y * (dR[0] + dR[8]) + z * (dR[5] + dR[7]) + w * (dR[2] - dR[6]));
  T dz = (x * (dR[2] + dR[6]) + y * (dR[5] + dR[7]) - T(2) * z * (dR[0] + dR[4]) + w * (dR[3] - dR[1]));
  T dw = (x * (dR[7] - dR[5]) + y * (dR[2] - dR[6]) + z * (dR[3] - dR[1]));

  T sum   = x * dx + y * dy + z * dz + w * dw;
  T scale = T(1) / max(x * x + y * y + z * z + w * w, T(1e-12));
  dq[0]   = T(2) * scale * (dx - x * sum * scale);
  dq[1]   = T(2) * scale * (dy - y * sum * scale);
  dq[2]   = T(2) * scale * (dz - z * sum * scale);
  dq[3]   = T(2) * scale * (dw - w * sum * scale);
}

template <typename T>
__global__ void quaternion_to_R_backward_kernel(
    int N, const T *__restrict__ q, const T *__restrict__ grad_R, T *__restrict__ grad_q) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    quaternion_to_R_backward_function<T>(q + idx * 4, grad_R + idx * 9, grad_q + idx * 4);
  }
}

Tensor quaternion_to_R_backward(Tensor &q, Tensor &grad_R) {
  grad_R        = grad_R.contiguous();
  int N         = q.numel() / 4;
  Tensor grad_q = torch::zeros_like(q);
  BCNN_ASSERT(grad_R.numel() == N * 9, "Error shape for grad_R");
  BCNN_ASSERT(grad_R.dtype() == q.dtype(), "Error dtype for grad_R");

  if (N == 0) return grad_q;
  if (q.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "quaternion_to_R_backward", [&] {
      quaternion_to_R_backward_kernel KERNEL_ARG(div_round_up(N, 256), 256)(N, q.contiguous().data_ptr<scalar_t>(),
          grad_R.contiguous().data_ptr<scalar_t>(), grad_q.data_ptr<scalar_t>());

      CHECK_CUDA_ERROR("quaternion_to_R_backward_kernel");
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "quaternion_to_R_backward", [&] {
      scalar_t *q_  = q.data_ptr<scalar_t>();
      scalar_t *dR_ = grad_R.data_ptr<scalar_t>();
      scalar_t *dq_ = grad_q.data_ptr<scalar_t>();
      for (int i = 0; i < N; i++) {
        quaternion_to_R_backward_function<scalar_t>(q_ + i * 4, dR_ + i * 9, dq_ + i * 4);
      }
    });
  }
  return grad_q;
}

REGIST_PYTORCH_EXTENSION(ops_3d_quaternion, {
  m.def("quaternion_to_R_forward", &quaternion_to_R_forward, "quaternion_to_R_forward (CUDA, CPU)");
  m.def("quaternion_to_R_backward", &quaternion_to_R_backward, "quaternion_to_R_backward (CUDA, CPU)");
})
}  // namespace OPS_3D