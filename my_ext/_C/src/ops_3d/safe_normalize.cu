#include "util.cuh"

template <typename T, int C = 2>
__global__ void safe_normalize_fwd_kernel(int N, float eps, const T* __restrict__ in, T* __restrict__ out) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  in    = in + n * C;
  out   = out + n * C;
  if (n < N) {
    float t = 0;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      t += in[c] * in[c];
    }
    t = t > eps ? t : eps;
    t = rsqrtf(t);
#pragma unroll
    for (int c = 0; c < C; ++c) {
      out[c] = in[c] * t;
    }
  }
}

template <typename T, int C = 2>
__global__ void safe_normalize_bwd_kernel(
    int N, float eps, const T* __restrict__ in, const T* __restrict__ grad_out, T* __restrict__ grad_in) {
  int n    = blockIdx.x * blockDim.x + threadIdx.x;
  in       = in + n * C;
  grad_out = grad_out + n * C;
  grad_in  = grad_in + n * C;
  if (n < N) {
    float t = 0, sum = 0;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      t += in[c] * in[c];
      sum += in[c];
    }
    t = rsqrtf(t > eps ? t : eps);
#pragma unroll
    for (int c = 0; c < C; ++c) {
      grad_in[c] = grad_out[c] * (t - t * t * t * in[c] * sum);
    }
  }
}

Tensor safe_normalize_forward(Tensor x, int dim = -1, float eps = 1e-20) {
  int C = x.size(-1);
  if (x.is_cuda() && 2 <= C && C <= 4 && dim == -1) {
    x        = x.contiguous();
    int N    = x.numel() / C;
    Tensor y = torch::empty_like(x);
    AT_DISPATCH_ALL_TYPES_AND_HALF(x.type(), "safe_normalize_fwd_kernel", [&] {
      switch (C) {
        case 2:
          safe_normalize_fwd_kernel<scalar_t, 2> KERNEL_ARG((N + 255) / 256, 256)(
              N, eps, x.data_ptr<scalar_t>(), y.data<scalar_t>());
          break;
        case 3:
          safe_normalize_fwd_kernel<scalar_t, 3> KERNEL_ARG((N + 255) / 256, 256)(
              N, eps, x.data_ptr<scalar_t>(), y.data<scalar_t>());
          break;
        case 4:
          safe_normalize_fwd_kernel<scalar_t, 4> KERNEL_ARG((N + 255) / 256, 256)(
              N, eps, x.data_ptr<scalar_t>(), y.data<scalar_t>());
          break;

        default: BCNN_ASSERT(false, "last dimension must be 2,3,4"); break;
      }
      CHECK_CUDA_ERROR("safe_normalize_fwd_kernel");
    });
    return y;
  } else {
    return x / torch::sqrt(torch::clamp_min(torch::sum(x * x, dim, true), eps));
  }
}

Tensor safe_normalize_backward(Tensor x, Tensor grad_out, int dim = -1, float eps = 1e-20) {
  int C = x.size(-1);
  if (x.is_cuda() && 2 <= C && C <= 4 && dim == -1) {
    x        = x.contiguous();
    int N    = x.numel() / C;
    Tensor y = torch::empty_like(x);
    AT_DISPATCH_ALL_TYPES_AND_HALF(x.type(), "safe_normalize_bwd_kernel", [&] {
      switch (C) {
        case 2:
          safe_normalize_bwd_kernel<scalar_t, 2> KERNEL_ARG((N + 255) / 256, 256)(
              N, eps, x.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(), y.data<scalar_t>());
          break;
        case 3:
          safe_normalize_bwd_kernel<scalar_t, 3> KERNEL_ARG((N + 255) / 256, 256)(
              N, eps, x.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(), y.data<scalar_t>());
          break;
        case 4:
          safe_normalize_bwd_kernel<scalar_t, 4> KERNEL_ARG((N + 255) / 256, 256)(
              N, eps, x.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(), y.data<scalar_t>());
          break;

        default: BCNN_ASSERT(false, "last dimension must be 2,3,4"); break;
      }
      CHECK_CUDA_ERROR("safe_normalize_bwd_kernel");
    });
    return y;
  } else {
    auto y = torch::rsqrt(torch::clamp_min(torch::sum(x * x, dim, true), eps));
    auto z = torch::sum(x, dim, true);
    return grad_out * (y - x * z * y * y * y);
  }
}

REGIST_PYTORCH_EXTENSION(ops_3d_safe_normalize, {
  m.def("safe_normalize_forward", &safe_normalize_forward, "safe_normalize_forward");
  m.def("safe_normalize_backward", &safe_normalize_backward, "safe_normalize_forward");
});