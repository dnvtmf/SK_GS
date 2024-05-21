#include "util.cuh"

template <typename T>
__global__ void cdist_top_match_label_kernel(int B, int N, int M, int D, const T* x, const T* y, const int64_t* lx,
    const int64_t* ly, T* dist, int64_t* index, bool largest) {
  int bid = blockIdx.x;
  int ns  = blockIdx.y;
  int tx  = threadIdx.x;
  int n   = ns * blockDim.x + tx;
  y       = y + bid * M * D;
  ly      = ly + bid * M;
  x       = x + (bid * N + n) * D;
  lx      = lx + n;
  dist    = dist + n;
  index   = index + n;

  T top    = 0;
  int topi = -1;
  if (n < N) {
    for (int m = 0; m < M; m++) {
      T sum = 0;
      for (int d = 0; d < D; ++d) {
        T temp = x[d] - y[m * D + d];
        sum += temp * temp;
      }
      if (lx[0] == ly[m] && (m < 0 || (sum > top) == largest)) {
        top  = sum;
        topi = m;
      }
    }
    *dist  = sqrt(top);
    *index = topi;
  }
}

vector<Tensor> cdist_top_match_label(
    const Tensor& x1, const Tensor& label1, const Tensor& x2, const Tensor& label2, bool largest = false) {
  int d = x1.ndimension();
  int N = x1.size(-2);
  int M = x2.size(-2);
  int D = x1.size(-1);
  int B = x1.numel() / N / D;

  Tensor distance = at::zeros(x1.sizes().slice(0, d - 1), x1.options());
  Tensor index    = at::zeros_like(distance, at::kLong);

  auto device = x1.device();

  BCNN_ASSERT(device == x2.device() && device == label1.device() && device == label2.device(),
      "all tensors must be same device");
  BCNN_ASSERT(x2.size(-1) == D, "The dimenstion of point must be ", D);
  BCNN_ASSERT(x2.numel() == B * M * D, "The shape[:-2] of x2 must be same with x1");
  BCNN_ASSERT(label1.sizes() == x1.sizes() && label2.sizes() == x2.sizes(), "label must have same shape with x");
  BCNN_ASSERT(label2.scalar_type() == at::kLong && label1.scalar_type() == at::kLong, "labels dtype != long");

  AT_DISPATCH_FLOATING_TYPES(x1.type(), "cdist top match label forward", [&] {
    const scalar_t* p1_ptr = x1.data<scalar_t>();
    const scalar_t* p2_ptr = x2.data<scalar_t>();
    const int64_t* l1_ptr  = label1.data<int64_t>();
    const int64_t* l2_ptr  = label2.data<int64_t>();
    scalar_t* dist_ptr     = distance.data<scalar_t>();
    int64_t* index_ptr     = index.data<int64_t>();
    if (x1.is_cuda()) {
      cdist_top_match_label_kernel<scalar_t> KERNEL_ARG(dim3((N - 1) / 128 + 1, B), 128)(
          B, N, M, D, p1_ptr, p2_ptr, l1_ptr, l2_ptr, dist_ptr, index_ptr, largest);
      CHECK_CUDA_ERROR("cdist_top_match_label_kernel");
    } else {
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          scalar_t top = 0;
          int topi     = -1;

          const scalar_t* p2_b_ptr = p2_ptr;
          for (int m = 0; m < M; m++) {
            if (*l1_ptr != l2_ptr[m]) continue;
            scalar_t sum = 0;
            for (int k = 0; k < D; ++k) {
              scalar_t t = p1_ptr[k] - p2_b_ptr[k];
              sum += t * t;
            }
            p2_b_ptr += D;
            if (topi < 0 || (sum > top) == largest) {
              top  = sum;
              topi = m;
            }
          }
          p1_ptr += D;
          *(dist_ptr++)  = sqrt(top);
          *(index_ptr++) = topi;
          l1_ptr++;
        }
        p2_ptr += M * D;
        l2_ptr += M;
      }
    }
  });
  return {distance, index};
}

REGIST_PYTORCH_EXTENSION(cdist_top_match_label, {
  m.def("cdist_top_match_label", &cdist_top_match_label, "Distance Top", py::arg("x1"), py::arg("label1"),
      py::arg("x2"), py::arg("label2"), py::arg("largest") = false);
})
