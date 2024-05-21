// adopt from https://github.com/KAIR-BAIR/nerfacc/blob/master/nerfacc/cuda/csrc/pdf.cu
#include "util.cuh"

__global__ void calc_new_pack_info(uint32_t num_rays, uint32_t total_old, bool merge, int* __restrict__ old_pack_info,
    int* __restrict__ num, int* __restrict__ new_pack_info) {
  __shared__ int total;
  int n, n_new, offset;
  total = 0;
  for (uint32_t i = 0; i < num_rays; i += blockDim.x) {
    int n = i + threadIdx.x;
    if (n < num_rays) {
      n_new = (old_pack_info == nullptr || old_pack_info[n * 2 + 1] > 0) ? num[n] : 0;
      n_new += (merge ? (old_pack_info ? old_pack_info[n * 2 + 1] : (total_old / num_rays)) : 0);
    } else
      n_new = 0;

    offset = scan_block<int>(n_new) + total;
    if (n < num_rays) {
      new_pack_info[n * 2 + 0] = offset - n_new;
      new_pack_info[n * 2 + 1] = n_new;
      // printf("new_pack_info[%d] = (%d, %d)\n", n, new_pack_info[n * 2 + 0], new_pack_info[n * 2 + 1]);
    }
    if (n + 1 == num_rays) {
      new_pack_info[num_rays * 2 + 0] = offset;
      new_pack_info[num_rays * 2 + 1] = 0;
    }
    if (threadIdx.x + 1 == blockDim.x) {
      // printf("set total from %d to %d\n", total, offset);
      total = offset;
    }
    __syncthreads();
  }
}

template <typename scalar_t>
inline __device__ uint32_t upper_bound(const scalar_t* data, uint32_t start, uint32_t end, const scalar_t val) {
  const uint32_t orig_start = start;
  while (start < end) {
    const uint32_t mid     = start + ((end - start) >> 1);
    const scalar_t mid_val = data[mid];
    if (!(mid_val > val))
      start = mid + 1;
    else
      end = mid;
  }
  return start;
}

inline __device__ uint32_t binary_search_chunk_id(
    const uint32_t item_id, const int32_t n_chunks, const int32_t* chunk_starts) {
  uint32_t start = 0;
  uint32_t end   = n_chunks;
  while (start < end) {
    const uint32_t mid     = start + ((end - start) >> 1);
    const uint32_t mid_val = chunk_starts[mid];
    if (!(mid_val > item_id))
      start = mid + 1;
    else
      end = mid;
  }
  return start;
}

template <typename T, typename T2>
__global__ void importance_sampling_kernel(const uint32_t num_rays, const uint32_t num_old, const uint32_t num_new,
    bool merge, T eps, const T* __restrict__ weights, const T* __restrict__ time_interval, const T* __restrict__ noise,
    const int* __restrict__ pack_info, const int* __restrict__ num, T* __restrict__ out_ts,
    int* __restrict__ ray_indices, int* __restrict__ new_pack_info) {
  // parallel per ray
  const uint32_t bid = blockIdx.x, tid = threadIdx.x;
  if (bid >= num_rays) return;

  uint32_t offset_o = pack_info ? pack_info[bid * 2 + 0] : bid * (num_old / num_rays);
  uint32_t number_o = pack_info ? pack_info[bid * 2 + 1] : (num_old / num_rays);
  uint32_t offset_n = new_pack_info ? new_pack_info[bid * 2 + 0] : bid * (num_new / num_rays);
  uint32_t number_n = num[bid];  // new_pack_info ? new_pack_info[bid * 2 + 1] : (num_new / num_rays);

  if (number_n == 0 || number_o == 0) return;
  weights += offset_o;
  time_interval += offset_o * 2;
  out_ts += offset_n * 2;
  ray_indices += offset_n * 2;
  const T* noise_last;
  if (noise != nullptr) {
    noise_last = noise + num_new;
    noise      = noise + offset_n;
  }

  __shared__ T2 cdfs[CUDA_NUM_THREADS];

  cdfs[tid] = scan_block<T2>((tid > 0 && tid <= number_o) ? weights[tid - 1] + eps : T(0));
  __syncthreads();
  cdfs[tid] = cdfs[tid] / cdfs[number_o];  // normalize
  __syncthreads();
  // if (bid == 0 && tid <= number_o) printf("cdf[%u] = %f\n", tid, cdfs[tid]);

  if (tid <= number_n) {
    T2 u = (tid + (noise ? T2(tid == number_n ? noise_last[bid] : noise[tid]) : (T2) 0.5)) / (number_n + 1);
    // search cdfs[p-1] <= u < cdfs[p]
    uint32_t p  = upper_bound<T2>(cdfs, 0, number_o, u);
    uint32_t p0 = clamp(p - 1, 0u, number_o), p1 = clamp(p, 0u, number_o);
    // if (bid == 0) printf("u[%u]=%f, in [%f, %f]\n", tid, u, cdfs[p0], cdfs[p1]);
    T2 t_lower = time_interval[p0 * 2 + 0];
    T2 t_upper = t_lower + time_interval[p0 * 2 + 1];
    T2 u_lower = cdfs[p0];
    T2 denom   = cdfs[p1] - u_lower;
    T2 t       = (u - u_lower) * (t_upper - t_lower) / (denom < eps ? T2(1.) : denom) + t_lower;
    if (tid < number_n) out_ts[tid * 2 + 0] = t;
    if (tid > 0) out_ts[(tid - 1) * 2 + 1] = t;
  }
  __syncthreads();
  if (tid < number_n) {
    out_ts[tid * 2 + 1] -= out_ts[tid * 2 + 0];
    ray_indices[tid * 2 + 0] = bid;
    ray_indices[tid * 2 + 1] = tid;
  }
}

vector<Tensor> importance_sample(Tensor weights, Tensor time_interval, at::optional<Tensor> pack_info, Tensor N,
    int max_old, int max_new, bool det = true, float eps = 1e-5, bool merge = false) {
  CHECK_INPUT(weights);
  CHECK_INPUT(time_interval);
  BCNN_ASSERT(weights.dtype() == time_interval.dtype(), "dtype for weights and time_interval should be same");
  CHECK_INPUT_AND_TYPE(N, torch::kInt32);

  uint32_t num_rays = N.numel(), total_old = weights.numel();

  BCNN_ASSERT(time_interval.numel() == total_old * 2, "Error shape for time_interval");
  if (pack_info.has_value()) {
    CHECK_INPUT_AND_TYPE(pack_info.value(), torch::kInt32);
    BCNN_ASSERT(pack_info.value().sizes() == at::IntArrayRef({num_rays + 1, 2}), "Error shape for pack_info");
  }

  Tensor new_info = torch::zeros({num_rays + 1, 2}, weights.options().dtype(torch::kInt32));
  calc_new_pack_info KERNEL_ARG(1, get_cuda_threads(num_rays))(num_rays, total_old, merge,
      pack_info.has_value() ? pack_info.value().data_ptr<int>() : nullptr, N.data_ptr<int>(), new_info.data_ptr<int>());

  CHECK_CUDA_ERROR("calc_new_pack_info");
  uint32_t total_new = new_info[-1][0].item<int>();
  Tensor noise;
  if (!det) noise = torch::rand({total_new + num_rays}, weights.options());
  Tensor ray_indices = torch::zeros({total_new, 2}, weights.options().dtype(torch::kInt32));
  Tensor ts          = torch::zeros({total_new, 2}, weights.options());
  Tensor cdf         = weights.reshape(-1).cumsum(-1);
  BCNN_ASSERT(max_old < CUDA_NUM_THREADS && max_new < CUDA_NUM_THREADS, "the number of samples per ray muse < 1024");

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(weights.scalar_type(), "importance_sample", ([&] {
    if constexpr (std::is_same<scalar_t, at::Half>::value) {
      importance_sampling_kernel<scalar_t, float> KERNEL_ARG(num_rays, get_cuda_threads(max(max_new, max_old) + 1))(
          num_rays, total_old, total_new, merge, (scalar_t) eps, weights.data_ptr<scalar_t>(),
          time_interval.data<scalar_t>(), det ? nullptr : noise.data_ptr<scalar_t>(),
          pack_info.has_value() ? pack_info.value().data_ptr<int>() : nullptr, N.data_ptr<int>(),
          ts.data_ptr<scalar_t>(), ray_indices.data_ptr<int>(), new_info.data_ptr<int>());
    } else {
      importance_sampling_kernel<scalar_t, scalar_t> KERNEL_ARG(num_rays, get_cuda_threads(max(max_new, max_old) + 1))(
          num_rays, total_old, total_new, merge, (scalar_t) eps, weights.data_ptr<scalar_t>(),
          time_interval.data<scalar_t>(), det ? nullptr : noise.data_ptr<scalar_t>(),
          pack_info.has_value() ? pack_info.value().data_ptr<int>() : nullptr, N.data_ptr<int>(),
          ts.data_ptr<scalar_t>(), ray_indices.data_ptr<int>(), new_info.data_ptr<int>());
    }
    CHECK_CUDA_ERROR("importance_sampling_kernel");
  }));
  return {ray_indices, ts, new_info};
}

REGIST_PYTORCH_EXTENSION(nerf_importance_sample, {
  m.def("importance_sample", &importance_sample, py::arg("weights"), py::arg("time_interval"), py::arg("packinfo"),
      py::arg("num_importance"), py::arg("max_old"), py::arg("max_new"), py::arg("det") = true, py::arg("eps") = 1e-5f,
      py::arg("merge") = false, "importance_sample (CUDA)");
});
