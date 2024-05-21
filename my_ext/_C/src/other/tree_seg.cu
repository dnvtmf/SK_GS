// open-set tree segment
#include "util.cuh"
namespace TreeSegment {

__device__ __host__ int find(int x, int* father) {
  if (x < 0) return x;
  return father[x] == x ? x : father[x] = find(father[x], father);
}
__device__ __host__ void union_set(int x, int y, int* father, int* count) {
  assert(x >= 0 && y >= 0);
  x = find(x, father);
  y = find(y, father);
  if (x == y) return;
  father[y] = x;
  count[x] += count[y];
}

__device__ __host__ bool in_same_set(int x, int y, int* father) { return find(x, father) == find(y, father); }

__global__ void tree_seg_2d_phase1_kernel(const int H, const int W, const int max_level,
    const float* __restrict__ depth, const float* __restrict__ common_right, const float* __restrict__ common_below,
    int* __restrict__ out, int* __restrict__ cnt) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int BX = blockDim.x, BY = blockDim.y;
  const int w = blockIdx.x * BX + threadIdx.x, h = blockIdx.y * BY + threadIdx.y;
  // initilzae
  if (h < H && w < W) {
    for (int l = 0; l > max_level; ++l) out[l * H * W + h * W + w] = l * H * W + h * W + w;
  }
  __syncthreads();
  // merge in a block
  if (h < H && w < W) {
    int level = min(int(round(depth[h * W + w])), max_level);
    int lr    = w + 1 == W ? 0 : min(level, int(round(min(depth[h * W + w + 1], common_right[h * (W - 1) + w]))));
    int lb    = h + 1 == H ? 0 : min(level, int(round(min(depth[h * W + w + W], common_below[h * W + w]))));
    for (int l = 0; l < level; ++l) {
      if (l < lr && tx + 1 < BX) union_set((l * H + h) * W + w, (l * H + h) * W + w + 1, out, cnt);
      if (l < lb && ty + 1 < BY) union_set((l * H + h) * W + w, (l * H + h + 1) * W + w, out, cnt);
    }
  }
}

Tensor tree_seg_2d(const Tensor& common_right, const Tensor& common_below, int max_level, int threshold) {
  BCNN_ASSERT(common_right.ndimension() == 2, "Error shape for common_right");
  CHECK_TYPE(common_right, torch::kFloat32);
  const int H = common_right.size(0);
  const int W = common_right.size(1) + 1;
  BCNN_ASSERT(common_below.sizes() == at::IntArrayRef({H - 1, W}), "Error shape for common_below");
  BCNN_ASSERT(common_right.dtype() == common_below.dtype(), "dtype must be same");
  BCNN_ASSERT(common_right.device() == common_below.device(), "device must be same");
  Tensor output = torch::zeros({max_level, H, W}, common_right.options().dtype(torch::kInt32));
  Tensor count  = torch::ones_like(output);

  auto com_r_ptr = common_right.contiguous().data_ptr<float>();
  auto com_b_ptr = common_below.contiguous().data_ptr<float>();
  auto out_ptr   = output.data<int>();
  auto cnt_ptr   = count.data<int>();
  BCNN_ASSERT(!common_right.is_cuda(), "Only support CPU");

  // initilaize
  for (int i = 0; i < max_level * H * W; ++i) out_ptr[i] = i;
  // merge
  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      int lr = w + 1 == W ? 0 : std::min(max_level, int(round(com_r_ptr[h * (W - 1) + w])));
      int lb = h + 1 == H ? 0 : std::min(max_level, int(round(com_b_ptr[h * W + w])));
      for (int l = 0; l < lr; ++l) union_set((l * H + h) * W + w, (l * H + h) * W + w + 1, out_ptr, cnt_ptr);
      for (int l = 0; l < lb; ++l) union_set((l * H + h) * W + w, (l * H + h + 1) * W + w, out_ptr, cnt_ptr);
    }
  }

  // reindex
  for (int i = 0; i < max_level * H * W; ++i) {
    int k      = find(i, out_ptr);
    out_ptr[i] = (k >= 0 && cnt_ptr[k] > threshold) ? k : -1;
  }
  return output;
}

vector<Tensor> tree_seg_3d(Tensor common_indices, Tensor common_depth, int max_level, int threshold) {
  // BCNN_ASSERT(depth.ndimension() == 1, "Error shape for depth");
  // const int N = depth.size(0);
  BCNN_ASSERT(common_indices.ndimension() == 2, "Error shape for common_indices");
  BCNN_ASSERT(common_indices.sizes() == common_depth.sizes(), "Error shape for common_depth");
  const int N = common_depth.size(0);
  const int B = common_indices.size(1);
  // CHECK_TYPE(depth, torch::kFloat32);
  CHECK_TYPE(common_depth, torch::kFloat32);
  CHECK_TYPE(common_indices, torch::kInt32);
  BCNN_ASSERT(common_depth.device() == common_indices.device(), "error device");

  Tensor indices = torch::zeros({max_level, N}, common_indices.options());
  Tensor parent  = torch::full_like(indices, -1);
  Tensor first   = torch::full_like(indices, -1);
  Tensor last    = torch::full_like(indices, -1);
  Tensor next    = torch::full_like(indices, -1);
  Tensor count   = torch::ones_like(indices);

  // const float* d_ptr  = depth.data_ptr<float>();
  const float* cd_ptr = common_depth.data_ptr<float>();
  const int* ci_ptr   = common_indices.data_ptr<int>();
  int* p_ptr          = parent.data<int>();
  int* f_ptr          = first.data<int>();
  int* l_ptr          = last.data<int>();
  int* n_ptr          = next.data<int>();
  int* t_ptr          = indices.data<int>();
  int* c_ptr          = count.data<int>();

  if (common_indices.is_cuda()) {
    BCNN_ASSERT(false, "CUDA is not supported now");
  }
  // init
  for (int i = 0; i < max_level * N; ++i) t_ptr[i] = i;
  // merge
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < B; ++j) {
      int k = ci_ptr[i * B + j];
      // int level = std::min(int(round(std::min(std::min(d_ptr[i], d_ptr[k]), cd_ptr[i * B + j]))), max_level);
      int level = std::min(int(round(cd_ptr[i * B + j])), max_level);
      int l;
      for (l = 0; l < level; ++l) {
        if (l > 0) {
          p_ptr[l * N + i]       = p_ptr[(l - 1) * N + i];
          f_ptr[(l - 1) * N + i] = f_ptr[l * N + i];
        }
        union_set(l * N + i, l * N + k, t_ptr, c_ptr);
      }
      // for (; l < max_level; ++l) t_ptr[l * N + i] = -1;
    }
  }
  // generate results
  for (int l = 0; l < max_level; ++l) {
    for (int i = 0; i < N; ++i) {
      int k = find(l * N + i, t_ptr);
      if (k < 0 || c_ptr[k] <= threshold) {
        t_ptr[l * N + i] = -1;
        continue;
      }
      t_ptr[l * N + i] = k;
      if (l > 0) p_ptr[l * N + i] = find(p_ptr[l * N + i], t_ptr);
      if (f_ptr[l * N + i] >= 0) f_ptr[l * N + i] = find(f_ptr[l * N + i], t_ptr);
      if (k != i) {  // insert l*N+i after k
        int next_i       = n_ptr[l * N + i];
        n_ptr[k]         = l * N + i;
        l_ptr[l * N + i] = k;
        if (next_i != -1) {
          n_ptr[l * N + i] = next_i;
          l_ptr[next_i]    = l * N + i;
        }
      }
    }
  }
  return {indices, parent, first, last, next};
}

REGIST_PYTORCH_EXTENSION(other_tree_seg, {
  m.def("tree_seg_2d", &tree_seg_2d, "tree_seg_2d (CPU)");
  m.def("tree_seg_3d", &tree_seg_3d, "tree_seg_3d (CPU)");
});
}  // namespace TreeSegment