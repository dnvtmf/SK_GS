#include "common.hpp"
#include "util.cuh"

namespace tree_seg {
using torch::Tensor;
#define DISPATCH_CASE_MY_TYPES(...)            \
  AT_DISPATCH_CASE(torch::kUInt8, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::kInt16, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::kInt32, __VA_ARGS__)

#define DISPATCH_MY_TYPES(TYPE, NAME, ...) AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_MY_TYPES(__VA_ARGS__))

template <typename T = int32_t>
__global__ void mask_to_binary_kernel(
    int W, const int32_t* __restrict__ start, const T* __restrict__ counts, bool* __restrict__ output) {
  auto* ptr     = counts + start[blockIdx.x];
  int e         = *ptr;
  bool v        = 0;
  bool* out_ptr = output + blockIdx.x * W;
  for (int x = threadIdx.x; x < W; x += blockDim.x) {
    while (e <= x) {
      e += *(++ptr);
      v ^= 1;
    }
    out_ptr[x] = v;
  }
}

Tensor mask_to_binary(int W, Tensor& start, Tensor& counts) {
  auto out_shape = start.sizes().vec();
  out_shape.push_back(W);
  Tensor out = torch::zeros(at::IntArrayRef(out_shape), torch::TensorOptions().dtype(at::kBool).device(start.device()));
  CHECK_CONTIGUOUS(start);
  CHECK_CONTIGUOUS(counts);
  BCNN_ASSERT(start.scalar_type() == torch::kInt32, "start must be int32");
  BCNN_ASSERT(start.device() == counts.device(), "start and counts must be same device");

  DISPATCH_MY_TYPES(counts.scalar_type(), "mask_to_binary", [&] {
    if (start.is_cuda()) {
      if (start.numel() > 0) {
        mask_to_binary_kernel<scalar_t> KERNEL_ARG(start.numel(), WARP_SIZE)(
            W, start.data_ptr<int32_t>(), counts.data_ptr<scalar_t>(), out.data<bool>());
        CHECK_CUDA_ERROR("mask_to_binary");
      }
    } else {
      bool* out_ptr            = out.data<bool>();
      const int32_t* start_ptr = start.data_ptr<int32_t>();
      for (int k = 0; k < start.numel(); k++) {
        int s       = start_ptr[k];
        int e       = k + 1 == start.numel() ? counts.numel() : start_ptr[k + 1];
        auto* c_ptr = counts.data_ptr<scalar_t>();
        bool* out_k = out_ptr + k * W;
        for (int j = s + 1, i = 0; j < e; j += 2) {
          i += c_ptr[j - 1];
          for (int c = 0; c < c_ptr[j]; ++c, ++i) {
            out_k[i] = true;
          }
        }
      }
    }
  });

  return out;
}

__global__ void mask_from_binary_kernel_1(int N, int W, const bool* __restrict__ masks, int* __restrict__ start) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  masks += x * W;
  if (x < N) {
    bool now = 0;
    int cnt  = 0;
    int num  = 0;
    for (int i = 0; i < W; ++i, masks++) {
      if (now == *masks)
        ++cnt;
      else {
        num += (cnt - 1) / 255 * 2 + 1;
        now ^= 1;
        cnt = 1;
      }
    }
    num += (cnt - 1) / 255 * 2 + 1 + now;
    start[x] = num;
  }
}

__global__ void mask_from_binary_kernel_2(
    int N, int W, const bool* __restrict__ masks, const int* __restrict__ start, uint8_t* __restrict__ counts) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  masks += x * W;
  // auto check = counts + start[x + 1 < N ? x + 1 : 0];
  counts += start[x];
  if (x < N) {
    bool now = 0;
    int cnt  = 0;
    int num  = 0;
    for (int i = 0; i < W; ++i, masks++) {
      if (now == *masks)
        ++cnt;
      else {
        while (cnt > 255) {
          *(counts++) = 255;
          *(counts++) = 0;
          cnt -= 255;
        }
        *(counts++) = cnt;
        now ^= 1;
        cnt = 1;
      }
    }
    while (cnt > 255) {
      *(counts++) = 255;
      *(counts++) = 0;
      cnt -= 255;
    }
    *(counts++) = cnt;
    if (now) *(counts++) = 0;
    // DEBUG
    // printf("x=%d, num right: %d", x, counts == check);
    // if (x + 1 < N) assert(counts == check);
  }
}

vector<Tensor> mask_from_binary(Tensor& masks) {
  CHECK_CONTIGUOUS(masks);
  BCNN_ASSERT(masks.ndimension() >= 2 && masks.scalar_type() == torch::kBool, "Error shape or dtype for masks");
  const int W = masks.size(-1);
  const int N = masks.numel() / W;
  Tensor start, counts;
  auto* ptr = masks.data_ptr<bool>();
  if (masks.is_cuda()) {
    auto shape = masks.sizes().vec();
    shape.pop_back();
    start = torch::zeros(N, masks.options().dtype(torch::kInt32));
    if (N > 0) {
      mask_from_binary_kernel_1 KERNEL_ARG((N + 255) / 256, 256)(N, W, ptr, start.data<int32_t>());
      CHECK_CUDA_ERROR("mask_from_binary_kernel_1");
    }
    int M  = start.sum().item<int>();
    start  = torch::cumsum(start, 0, torch::kInt32) - start;
    counts = torch::zeros(M, masks.options().dtype(torch::kUInt8));
    if (N > 0) {
      mask_from_binary_kernel_2 KERNEL_ARG((N + 255) / 256, 256)(
          N, W, ptr, start.data_ptr<int32_t>(), counts.data<uint8_t>());
      CHECK_CUDA_ERROR("mask_from_binary_kernel_2");
    }
    start = start.view(shape);
  } else {
    vector<int> s;
    vector<uint8_t> c;
    for (int i = 0; i < N; ++i) {
      s.push_back(c.size());
      bool now = 0;
      int cnt  = 0;
      for (int j = 0; j < W; ++j, ptr++) {
        if (now == *ptr)
          cnt++;
        else {
          while (cnt > 255) {
            c.push_back(255);
            c.push_back(0);
            cnt -= 255;
          }
          c.push_back(cnt);
          now = *ptr;
          cnt = 1;
        }
      }
      while (cnt > 255) {
        c.push_back(255);
        c.push_back(0);
        cnt -= 255;
      }
      c.push_back(cnt);
      if (now) c.push_back(0);
    }
    start  = torch::tensor(s, torch::kInt32);
    counts = torch::tensor(c, torch::kUInt8);
  }
  return {start, counts};
}

template <typename T = int32_t>
__global__ void intersect_cuda(int N, int M, int H, int W, const int32_t* __restrict__ a_start,
    const T* __restrict__ a_counts, const int32_t* __restrict__ b_start, const T* __restrict__ b_counts,
    float_t* __restrict__ out) {
  const int i = blockIdx.x / M, j = blockIdx.x % M;
  float sum = 0;
  for (int k = threadIdx.x; k < H; k += blockDim.x) {
    auto *ca = a_counts + a_start[i * H + k], *cb = b_counts + b_start[j * H + k];
    int na = *ca, nb = *cb;
    bool a = 0, b = 0;
    while (na < W || nb < W) {
      if (na < nb)
        a ^= 1, na += *(++ca);
      else
        b ^= 1, nb += *(++cb);
      if (a & b) sum += min(na, nb) - max(na - *ca, nb - *cb);
    }
  }
  __syncthreads();
  reduce_sum_block<float, false>(sum);
  if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

template <typename T = int32_t>
void intersect_cpu(int N, int M, int H, int W, const int32_t* a_start, const T* a_counts, const int32_t* b_start,
    const T* b_counts, float_t* out) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      float_t sum = 0;
      for (int k = 0; k < H; ++k) {
        int sa = a_start[i * H + k], sb = b_start[j * H + k];
        auto *ca = a_counts + sa, *cb = b_counts + sb;
        int na = *ca, nb = *cb;
        bool a = 0, b = 0;
        while (na < W || nb < W) {
          if (na < nb) {
            a ^= 1;
            ca++;
            na += *ca;
          } else {
            b ^= 1;
            cb++;
            nb += *cb;
          }
          if (a & b) sum += min(na, nb) - max(na - *ca, nb - *cb);
        }
      }
      *(out++) = sum;
    }
  }
}

void intersect(int W, Tensor& a_start, Tensor& a_counts, Tensor& b_start, Tensor& b_counts, Tensor& output) {
  BCNN_ASSERT(a_start.device() == b_start.device(), "a, b must be same device");
  BCNN_ASSERT(a_start.scalar_type() == at::kInt && b_start.scalar_type() == at::kInt, "dtype of a, b must be int32");
  BCNN_ASSERT(a_start.size(-1) == b_start.size(-1), "a, b must have same (H, W)");
  BCNN_ASSERT(a_counts.dtype() == a_counts.dtype(), "a_counts, b_counts must have same dtype");
  const int H = a_start.size(-1);
  const int N = a_start.numel() / H, M = b_start.numel() / H;
  BCNN_ASSERT(output.numel() == N * M && output.scalar_type() == at::kFloat, "Error output shape or dtype");
  DISPATCH_MY_TYPES(a_counts.type(), "intersect", [&] {
    if (a_start.is_cuda()) {
      intersect_cuda KERNEL_ARG(N * M, min(1024, get_cuda_threads(H)))(N, M, H, W, a_start.data_ptr<int32_t>(),
          a_counts.data_ptr<scalar_t>(), b_start.data_ptr<int32_t>(), b_counts.data_ptr<scalar_t>(),
          output.data<float>());
      CHECK_CUDA_ERROR("intersect");
    } else {
      intersect_cpu(N, M, H, W, a_start.data_ptr<int32_t>(), a_counts.data_ptr<scalar_t>(), b_start.data_ptr<int32_t>(),
          b_counts.data_ptr<scalar_t>(), output.data<float>());
    }
  });
}

}  // namespace tree_seg
REGIST_PYTORCH_EXTENSION(tree_seg_mask, {
  m.def("mask_to_binary", &tree_seg::mask_to_binary, "mask_to_binary");
  m.def("mask_from_binary", &tree_seg::mask_from_binary, "mask_from_binary");
  m.def("intersect", &tree_seg::intersect, "intersect");
});