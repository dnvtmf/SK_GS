#include "util.cuh"

template <typename T>
__global__ void cdist_top_kernel(
    int B, int N, int M, int D, const T* x, const T* y, T* dist, int64_t* index, bool largest) {
  int bid = blockIdx.y;
  int ns  = blockIdx.x;
  int tx  = threadIdx.x;
  int n   = ns * blockDim.x + tx;
  y       = y + bid * M * D;
  x       = x + (bid * N + n) * D;
  dist    = dist + bid * N + n;
  index   = index + bid * N + n;

  T top    = 0;
  int topi = 0;
  if (n < N) {
    for (int m = 0; m < M; m++) {
      T sum = 0;
      for (int d = 0; d < D; ++d) {
        T temp = x[d] - y[m * D + d];
        sum += temp * temp;
      }
      if (m == 0 || (sum > top) == largest) {
        top  = sum;
        topi = m;
      }
    }
    *dist  = sqrt(top);
    *index = topi;
  }
}

vector<Tensor> cdist_top(const Tensor x1, const Tensor x2, bool largest = false) {
  int d = x1.ndimension();
  int N = x1.size(-2);
  int M = x2.size(-2);
  int D = x1.size(-1);
  int B = x1.numel() / N / D;

  Tensor distance = at::zeros(x1.sizes().slice(0, d - 1), x1.options());
  Tensor index    = at::zeros_like(distance, at::kLong);

  BCNN_ASSERT(x1.device() == x2.device(), "x1 and x2 must be same device");
  BCNN_ASSERT(x2.size(-1) == D, "The dimenstion of point must be ", D);
  BCNN_ASSERT(x2.numel() == B * M * D, "The shape[:-2] of x2 must be same with x1");

  AT_DISPATCH_FLOATING_TYPES(x1.type(), "cdist top  forward", [&] {
    const scalar_t* p1_ptr = x1.data<scalar_t>();
    const scalar_t* p2_ptr = x2.data<scalar_t>();
    scalar_t* dist_ptr     = distance.data<scalar_t>();
    int64_t* index_ptr     = index.data<int64_t>();
    if (x1.is_cuda()) {
      cdist_top_kernel<scalar_t> KERNEL_ARG(dim3((N - 1) / 128 + 1, B), 128)(
          B, N, M, D, p1_ptr, p2_ptr, dist_ptr, index_ptr, largest);
      CHECK_CUDA_ERROR("cdist_top_kernel");
    } else {
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          scalar_t best            = 0;
          int besti                = 0;
          const scalar_t* p2_b_ptr = p2_ptr;
          for (int m = 0; m < M; m++) {
            scalar_t sum = 0;
            for (int k = 0; k < D; ++k) {
              scalar_t t = p1_ptr[k] - p2_b_ptr[k];
              sum += t * t;
            }
            p2_b_ptr += D;
            if (m == 0 || (sum > best) == largest) {
              best  = sum;
              besti = m;
            }
          }
          p1_ptr += D;
          *(dist_ptr++)  = sqrt(best);
          *(index_ptr++) = besti;
        }
        p2_ptr += M * D;
      }
    }
  });
  return {distance, index};
}

template <typename T>
__global__ void distance_top_backward_kernel(int B, int N, int M, int D, const T* p1, const T* p2, const T* grad_dist,
    const T* dist, const int64_t* index, T* g1, T* g2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * N) {
    int j2 = index[idx];
    int b  = idx / N;
    p1     = p1 + idx * D;
    p2     = p2 + (b * M + j2) * D;
    g1     = g1 + idx * D;
    g2     = g2 + (b * M + j2) * D;
    T g    = grad_dist[idx] / dist[idx];

    for (int d = 0; d < D; ++d) {
      T t   = g * (p1[d] - p2[d]);
      g1[d] = t;
      atomicAdd(g2 + d, -t);
    }
  }
}

vector<Tensor> distance_top_backward(
    const Tensor x1, const Tensor x2, const Tensor distance, const Tensor index, const Tensor grad_dist) {
  int N         = x1.size(-2);
  int D         = x1.size(-1);
  int B         = x1.numel() / N / D;
  int M         = x2.size(-2);
  Tensor grad_1 = at::zeros_like(x1);
  Tensor grad_2 = at::zeros_like(x2);

  AT_DISPATCH_FLOATING_TYPES(x1.type(), "cdist top  backward", [&] {
    const scalar_t* p1_ptr = x1.data<scalar_t>();
    const scalar_t* p2_ptr = x2.data<scalar_t>();
    const scalar_t* d_ptr  = distance.data<scalar_t>();
    const int64_t* idx_ptr = index.data<int64_t>();
    const scalar_t* gd_ptr = grad_dist.data<scalar_t>();
    scalar_t* g1_ptr       = grad_1.data<scalar_t>();
    scalar_t* g2_ptr       = grad_2.data<scalar_t>();

    if (x1.is_cuda()) {
      distance_top_backward_kernel<scalar_t> KERNEL_ARG((B * N - 1) / 256 + 1, 256)(
          B, N, M, D, p1_ptr, p2_ptr, gd_ptr, d_ptr, idx_ptr, g1_ptr, g2_ptr);
      CHECK_CUDA_ERROR("distance_top_backward_kernel");
    } else {
      for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {
          int j2                    = *idx_ptr;
          scalar_t g                = *gd_ptr / *d_ptr;
          const scalar_t* p2_ij_ptr = p2_ptr + j2 * D;
          scalar_t* g2_ij_ptr       = g2_ptr + j2 * D;
          for (int d = 0; d < D; ++d) {
            scalar_t t = g * (p1_ptr[d] - p2_ij_ptr[d]);
            g1_ptr[d]  = t;
            g2_ij_ptr[d] -= t;
          }
          idx_ptr++;
          gd_ptr++;
          d_ptr++;
          p1_ptr += D;
          g1_ptr += D;
        }
        p2_ptr += M * D;
        g2_ptr += M * D;
      }
    }
  });

  return {grad_1, grad_2};
}

REGIST_PYTORCH_EXTENSION(dinstance_top, {
  m.def("cdist_top", &cdist_top, "Distance Top", py::arg("x1"), py::arg("x2"), py::arg("largest") = false);
  m.def("cdist_top_backward", &distance_top_backward, "Distance Top backward");
})
