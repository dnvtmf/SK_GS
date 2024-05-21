#include "util.cuh"
namespace SPNet {

/*
input: xyz (b, n, 3) new_xyz (b, m, 3)
output: idx (b, m, nsample) dist2 (b, m, nsample)

Note: n is the number of points, m is the number of clusters
Note: nsample <= 1000

input: idx_c (b, n, ks) lab (b, n, 1) cid (b, m, 1)
output: idx (b, m, n) cnt (b, m, 1) clab (b, m, class)
*/
__global__ void assomatrix_label_cuda_kernel(int b, int n, int m, int ks, int category, const int *__restrict__ idx_c,
    const int *__restrict__ lab, const int *__restrict__ cid, int *__restrict__ idx, int *__restrict__ cnt,
    int *__restrict__ clab) {
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= m) return;

  // new_xyz += bs_idx * m * 3 + pt_idx * 3;
  // cid += bs_idx * m * 1 + pt_idx * 1;
  cid += bs_idx * m * 1 + pt_idx * 1;
  // xyz += bs_idx * n * 3;
  idx_c += bs_idx * n * ks;  // add

  idx += bs_idx * m * n + pt_idx * n;
  cnt += bs_idx * m * 1 + pt_idx * 1;  // count number of points located in one superpoint
  if (lab) {
    lab += bs_idx * n * 1;  // add
    clab += bs_idx * m * category + pt_idx * category;
  }
  int cluster_id = cid[0];

  int num = 0;
  for (int k = 0; k < n; k++) {
    int k_lab = lab ? lab[k] : 0;
    for (int j = 0; j < ks; j++) {
      int id = idx_c[k * ks + j];  // cluster id of i-th point
      if (id == cluster_id) {
        idx[k] = 1;
        if (lab) clab[k_lab]++;
        num++;
      }
    }
  }
  cnt[0] = num;
}

void assomatrix_label_cuda(int b, int n, int m, int ks, int category, Tensor idx_c_tensor, Tensor lab_tensor,
    Tensor cid_tensor, Tensor idx_tensor, Tensor cnt_tensor, Tensor clab_tensor) {
  CHECK_INPUT(idx_c_tensor);
  CHECK_INPUT(cid_tensor);

  const int *idx_c = idx_c_tensor.data<int>();
  const int *lab   = lab_tensor.data<int>();
  const int *cid   = cid_tensor.data<int>();
  int *idx         = idx_tensor.data<int>();
  int *cnt         = cnt_tensor.data<int>();
  int *clab        = clab_tensor.data<int>();

  dim3 blocks(div_round_up(m, 256), b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(256);
  // param new_xyz: (B, m, 3)
  // param xyz: (B, n, 3)
  // param idx: (B, m, n)
  // param cnt: (B, m, 1)
  // param clab: (B, m, class)
  assomatrix_label_cuda_kernel KERNEL_ARG(blocks, threads)(b, n, m, ks, category, idx_c, lab, cid, idx, cnt, clab);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  CHECK_CUDA_ERROR("assomatrix_label_cuda_kernel");
}

// input: val_c (b, n, ks) idx_c (b, n, ks) cid (b, m, 1)
// output: idx (b, m, n) cnt (b, m, 1)
__global__ void assomatrix_float_cuda_kernel(int b, int n, int m, int ks, const float *__restrict__ val_c,
    const int *__restrict__ idx_c, const int *__restrict__ cid, float *__restrict__ idx, int *__restrict__ cnt) {
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= m) return;

  cid += bs_idx * m * 1 + pt_idx * 1;
  val_c += bs_idx * n * ks;  // add
  idx_c += bs_idx * n * ks;  // add
  idx += bs_idx * m * n + pt_idx * n;
  cnt += bs_idx * m * 1 + pt_idx * 1;  // count number of points located in one superpoint

  int cluster_id = cid[0];

  int num = 0;
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < ks; j++) {
      int id    = idx_c[k * ks + j];  // cluster id of i-th point
      float val = val_c[k * ks + j];
      if (id == cluster_id) {
        // tmpi[k] = val;
        idx[k] = val;
        num++;
      }
    }
  }
  cnt[0] = num;
}

template <typename T>
__global__ void SPNet_sp_scatter_features_kernel(int N, int M, int K, int F, const T *__restrict__ feat_p,
    const T *__restrict__ asso_matrix, const int64_t *__restrict__ idx_c2p, T *__restrict__ feat_sp,
    T *__restrict__ weight_sp) {
  /*
  N: number of points, may be 1e4~1e6;
  M: number of superponits, may be 100~1000;
  K: K-nearest superpoint for each ponts, may be 2-10
  F: the dimenstion of features, may be 4-256
  feat_sp[j, :] = sum [i=1->N]{ [j in KNN_i] * asso_matrix[i, j] * feat_p[i, :]}
  weight_sp[j] = sum [i=1->N]{ [j in KNN_i] * asso_matrix[i, j]
  */
  const int n  = blockDim.x * blockIdx.x + threadIdx.x;
  const int sp = blockIdx.y;
  if (n >= N || sp >= M) return;
  // const int tid = threadIdx.x;
  feat_p += n * F;
  asso_matrix += n * K;
  idx_c2p += n * K;
  for (int k = 0; k < K; ++k) {
    if (idx_c2p[k] == sp) {
      for (int f = 0; f < F; ++f) {
        atomicAdd(&feat_sp[sp * F + f], asso_matrix[k] * feat_p[f]);
      }
      atomicAdd(&weight_sp[sp], asso_matrix[k]);
    }
  }
}

Tensor SPNet_superpoint_features(Tensor &feat_p, Tensor &neighbor, Tensor &G, int M) {
  CHECK_CUDA_AND_TYPE(feat_p, torch::kFloat32);
  CHECK_CUDA_AND_TYPE(G, torch::kFloat32);
  CHECK_CUDA_AND_TYPE(neighbor, torch::kInt64);
  CHECK_NDIM(feat_p, 2);
  int N = feat_p.size(0);
  int F = feat_p.size(1);
  CHECK_NDIM(neighbor, 2);
  int K = neighbor.size(1);
  BCNN_ASSERT(neighbor.size(0) == N, "Error shape for neighbor");
  CHECK_SHAPE(G, {N, K});

  Tensor feat_sp = torch::zeros({M, F}, feat_p.options());
  Tensor weight  = torch::zeros({M, 1}, feat_p.options());
  SPNet_sp_scatter_features_kernel KERNEL_ARG(dim3(div_round_up(N, 256), M), 256)(N, M, K, F,
      feat_p.contiguous().data_ptr<float>(), G.contiguous().data_ptr<float>(),
      neighbor.contiguous().data_ptr<int64_t>(), feat_sp.data_ptr<float>(), weight.data_ptr<float>());
  CHECK_CUDA_ERROR("SPNet_sp_scatter_features_kernel");

  return feat_sp / weight.clamp_min(1e-5);
}

REGIST_PYTORCH_EXTENSION(other_SPNet,
    { m.def("SPNet_superpoint_features", &SPNet_superpoint_features, "SPNet_superpoint_features (CUDA)"); });
}  // namespace SPNet