#include "glm_helper.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <type_traits>
#include "ops_3d_types.h"
#include "ops_3d.h"
#include "util.cuh"

namespace OPS_3D {

template <typename T>
void __global__ perspective_kernel(
    int N, const T* __restrict__ fovy, T aspect_r, T m22, T m23, mat4<T>* __restrict__ out) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    T c            = 1. / tan(fovy[idx] * (T) 0.5);
    out[idx](0, 0) = c * aspect_r;
    out[idx](1, 1) = c;
    out[idx](2, 2) = m22;
    out[idx](2, 3) = m23;
    out[idx](3, 2) = T(-1);
  }
}

Tensor perspective(Tensor fovy, double aspect, double near, double far) {
  fovy       = fovy.contiguous();
  auto shape = fovy.sizes().vec();
  shape.push_back(4);
  shape.push_back(4);
  Tensor matrix = torch::zeros(shape, fovy.options());
  int N         = fovy.numel();

  if (fovy.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(fovy.scalar_type(), "perspective", [&] {
      double m22 = (near + far) / (near - far);
      double m23 = (2 * near * far) / (near - far);
      perspective_kernel<scalar_t> KERNEL_ARG(div_round_up(N, 256), 256)(
          N, fovy.data_ptr<scalar_t>(), 1. / aspect, m22, m23, (mat4<scalar_t>*) matrix.data<scalar_t>());
      CHECK_CUDA_ERROR("perspective_kernel");
    });
  }
  if (fovy.is_cpu()) {
    AT_DISPATCH_FLOATING_TYPES(fovy.scalar_type(), "perspective", [&] {
      auto in_ptr  = fovy.data_ptr<scalar_t>();
      auto out_ptr = matrix.data_ptr<scalar_t>();
      for (int i = 0; i < N; ++i) {
        auto m_i = glm::perspective<scalar_t>(in_ptr[i], (scalar_t) aspect, (scalar_t) near, (scalar_t) far);
        mat4_copy_to<scalar_t>(m_i, out_ptr + i * 16);
      }
    });
  }

  return matrix;
}

template <typename T>
void __global__ ortho_kernel(int N, const T* __restrict__ box, mat4<T>* __restrict__ out) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    box = box + idx * 6;
    T l = box[0], r = box[1], b = box[2], t = box[3], n = box[4], f = box[5];
    out[idx](0, 0) = 2 / (r - l);
    out[idx](1, 1) = 2 / (t - b);
    out[idx](0, 3) = -(r + l) / (r - l);
    out[idx](1, 3) = -(t + b) / (t - b);
    out[idx](2, 2) = -2 / (f - n);
    out[idx](2, 3) = -(f + n) / (f - n);
    out[idx](3, 3) = 1;
  }
}

Tensor ortho(Tensor box) {
  box        = box.contiguous();
  auto shape = box.sizes().vec();
  BCNN_ASSERT(shape.back() == 6, "ERROR shape of input");
  shape.pop_back();
  shape.push_back(4);
  shape.push_back(4);
  Tensor matrix = torch::zeros(shape, box.options());
  int N         = box.numel() / 6;

  if (box.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(box.scalar_type(), "ortho", [&] {
      ortho_kernel<scalar_t> KERNEL_ARG(div_round_up(N, 256), 256)(
          N, box.data_ptr<scalar_t>(), (mat4<scalar_t>*) matrix.data<scalar_t>());
      CHECK_CUDA_ERROR("ortho_kernel");
    });
  }
  if (box.is_cpu()) {
    AT_DISPATCH_FLOATING_TYPES(box.scalar_type(), "ortho", [&] {
      auto in_ptr  = box.data_ptr<scalar_t>();
      auto out_ptr = matrix.data_ptr<scalar_t>();
      for (int i = 0; i < N; ++i) {
        auto in_i = in_ptr + i * 6;
        auto m_i  = glm::ortho<scalar_t>(in_i[0], in_i[1], in_i[2], in_i[3], in_i[4], in_i[5]);
        mat4_copy_to<scalar_t>(m_i, out_ptr + i * 16);
      }
    });
  }

  return matrix;
}
REGIST_PYTORCH_EXTENSION(ops_3d_coordinate, {
  m.def("perspective", &perspective, "perspective (CUDA, CPU)");
  m.def("ortho", &ortho, "ortho (CUDA, CPU)");
})
}  // namespace OPS_3D