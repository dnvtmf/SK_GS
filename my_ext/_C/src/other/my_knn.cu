#include "util.cuh"

__forceinline__ __device__ float _dist(float3& a, float3& b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

void __global__ my_knn_update_kernel(
    int N, int K, int* __restrict__ indices, float* __restrict__ dist2, float3* __restrict__ points) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float3 now = points[n];
  extern __shared__ int smem[];
  int* knn_index   = smem + threadIdx.x * K;
  float* knn_dist2 = (float*) &smem[(threadIdx.x + blockDim.x) * K];
  // init from last neighborhoods
  for (int k = 0; k < K; ++k) {
    int x   = indices[n * K + k];
    float d = _dist(now, points[x]);
    for (int i = 0; i < k; ++i) {
      if (d <= knn_dist2[i]) {
        _swap(x, knn_index[i]);
        _swap(d, knn_dist2[i]);
      }
    }
    knn_index[k] = x;
    knn_dist2[k] = d;
  }
  // update from neighborhoods of neighborhoods
  for (int k1 = 0; k1 < K; ++k1) {
    int x = indices[n * K + k1];
    for (int k2 = 0; k2 < K; ++k2) {
      int y   = indices[x * K + k2];
      float d = _dist(now, points[y]);
      for (int i = 0; i < K; ++i) {
        if (y == knn_index[i]) break;
        if (d <= knn_dist2[i]) _swap(d, knn_dist2[i]), _swap(y, knn_index[i]);
      }
    }
  }
  // write results
  for (int i = 0; i < K; ++i) {
    indices[n * K + i] = knn_index[i];
    if (dist2) dist2[n * K + i] = knn_dist2[i];
  }
}

void my_knn_update(Tensor& index, at::optional<Tensor>& distance, Tensor& points) {
  CHECK_INPUT_AND_TYPE(index, torch::kInt32);
  CHECK_INPUT_AND_TYPE(points, torch::kFloat32);
  CHECK_NDIM(index, 2);
  int N = index.size(0);
  int K = index.size(1);
  if (distance.has_value()) {
    CHECK_INPUT_AND_TYPE(distance.value(), torch::kFloat32);
    CHECK_SHAPE(distance.value(), {N, K});
  }
  CHECK_SHAPE(index, N, K);
  CHECK_SHAPE(points, N, 3);

  my_knn_update_kernel KERNEL_ARG(div_round_up(N, 256), 256, 256 * K * (sizeof(int) + sizeof(float)))(N, K,
      index.data_ptr<int>(), distance.has_value() ? distance.value().data_ptr<float>() : nullptr,
      (float3*) points.data_ptr<float>());
  CHECK_CUDA_ERROR("my_knn_update_kernel");
}

REGIST_PYTORCH_EXTENSION(other_my_knn, { m.def("my_knn_update", &my_knn_update, "my_knn_update (CUDA)"); });