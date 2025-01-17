#include "util.cuh"

namespace FarthesPointSampling {
// ------------------------------ gathering  -----------------------------------
// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
__global__ void gathering_forward_cuda_kernel(
    int b, int c, int n, int m, const float *points, const int *idx, float *out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a                    = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
__global__ void gathering_backward_cuda_kernel(
    int b, int c, int n, int m, const float *grad_out, const int *idx, float *grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        atomicAdd(grad_points + (i * c + l) * n + a, grad_out[(i * c + l) * m + j]);
      }
    }
  }
}

void gathering_forward_cuda_launcher(int b, int c, int n, int m, const float *points, const int *idx, float *out) {
  gathering_forward_cuda_kernel KERNEL_ARG(dim3(b, c, 1), get_cuda_threads(m), 0)(b, c, n, m, points, idx, out);
}

void gathering_backward_cuda_launcher(
    int b, int c, int n, int m, const float *grad_out, const int *idx, float *grad_points) {
  gathering_backward_cuda_kernel KERNEL_ARG(dim3(b, c, 1), get_cuda_threads(m), 0)(
      b, c, n, m, grad_out, idx, grad_points);
}

// ------------------------------ gathering int -----------------------------------
// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
__global__ void gathering_int_forward_cuda_kernel(
    int b, int c, int n, int m, const int *points, const int *idx, int *out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a                    = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
__global__ void gathering_int_backward_cuda_kernel(
    int b, int c, int n, int m, const float *grad_out, const int *idx, float *grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int a = idx[i * m + j];
        atomicAdd(grad_points + (i * c + l) * n + a, grad_out[(i * c + l) * m + j]);
      }
    }
  }
}

void gathering_int_forward_cuda_launcher(int b, int c, int n, int m, const int *points, const int *idx, int *out) {
  gathering_int_forward_cuda_kernel KERNEL_ARG(dim3(b, c, 1), get_cuda_threads(m), 0)(b, c, n, m, points, idx, out);
}

void gathering_int_backward_cuda_launcher(
    int b, int c, int n, int m, const float *grad_out, const int *idx, float *grad_points) {
  gathering_int_backward_cuda_kernel KERNEL_ARG(dim3(b, c, 1), get_cuda_threads(m), 0)(
      b, c, n, m, grad_out, idx, grad_points);
}

// ------------------------------ gathering cluster -----------------------------------
// input: points(b, c, n) idx(b, m) idx_3d(b, m, k)
// output: out(b, c, m)
__global__ void gathering_cluster_forward_cuda_kernel(
    int b, int c, int n, int m, int k, const float *points, const int *idx, const int *idx_3d, float *out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int tmp                  = idx[i * m + j];               // add
        int a                    = idx_3d[i * m + j * k + tmp];  // add
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

// input: grad_out(b, c, m) idx(b, m) idx_3d(b, m, k)
// output: grad_points(b, c, n)
__global__ void gathering_cluster_backward_cuda_kernel(
    int b, int c, int n, int m, int k, const float *grad_out, const int *idx, const int *idx_3d, float *grad_points) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int tmp = idx[i * m + j];               // add
        int a   = idx_3d[i * m + j * k + tmp];  // add
        atomicAdd(grad_points + (i * c + l) * n + a, grad_out[(i * c + l) * m + j]);
      }
    }
  }
}

void gathering_cluster_forward_cuda_launcher(
    int b, int c, int n, int m, int k, const float *points, const int *idx, const int *idx_3d, float *out) {
  gathering_cluster_forward_cuda_kernel KERNEL_ARG(dim3(b, c, 1), get_cuda_threads(m), 0)(
      b, c, n, m, k, points, idx, idx_3d, out);
}

void gathering_cluster_backward_cuda_launcher(
    int b, int c, int n, int m, int k, const float *grad_out, const int *idx, const int *idx_3d, float *grad_points) {
  gathering_cluster_backward_cuda_kernel KERNEL_ARG(dim3(b, c, 1), get_cuda_threads(m), 0)(
      b, c, n, m, k, grad_out, idx, idx_3d, grad_points);
}

__device__ void __update(float *dists, int *dists_i, int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1]   = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void furthestsampling_cuda_kernel(int b, int n, int m, const float *dataset, float *temp, int *idxs) {
  if (m <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  temp += batch_index * n;
  idxs += batch_index * m;
  int tid          = threadIdx.x;
  const int stride = block_size;
  int old          = 0;
  if (threadIdx.x == 0) idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti  = 0;
    float best = -1;
    float x1   = dataset[old * 3 + 0];
    float y1   = dataset[old * 3 + 1];
    float z1   = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      // if (mag <= 1e-3)
      //     continue;
      float d  = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      float d2 = min(d, temp[k]);
      temp[k]  = d2;
      besti    = d2 > best ? k : besti;
      best     = d2 > best ? d2 : best;
    }
    dists[tid]   = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
      if (tid < 512) {
        __update(dists, dists_i, tid, tid + 512);
      }
      __syncthreads();
    }
    if (block_size >= 512) {
      if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}

void furthestsampling_cuda_launcher(int b, int n, int m, const float *dataset, float *temp, int *idxs) {
  unsigned int n_threads = get_cuda_threads(n);
  switch (n_threads) {
    case 1024: furthestsampling_cuda_kernel<1024> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    case 512: furthestsampling_cuda_kernel<512> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    case 256: furthestsampling_cuda_kernel<256> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    case 128: furthestsampling_cuda_kernel<128> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    case 64: furthestsampling_cuda_kernel<64> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    case 32: furthestsampling_cuda_kernel<32> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    case 16: furthestsampling_cuda_kernel<16> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    case 8: furthestsampling_cuda_kernel<8> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    case 4: furthestsampling_cuda_kernel<4> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    case 2: furthestsampling_cuda_kernel<2> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    case 1: furthestsampling_cuda_kernel<1> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs); break;
    default: furthestsampling_cuda_kernel<512> KERNEL_ARG(b, n_threads, 0)(b, n, m, dataset, temp, idxs);
  }
}

Tensor FurthestSampling(Tensor &points, int M) {
  CHECK_CUDA(points);
  CHECK_TYPE(points, torch::kFloat32);
  CHECK_NDIM(points, 3);
  int B       = points.size(0);
  int N       = points.size(1);
  Tensor temp = torch::full({B, N}, 1e10, points.options());
  Tensor idx  = torch::zeros({B, M}, points.options().dtype(torch::kInt32));
  furthestsampling_cuda_launcher(
      B, N, M, points.contiguous().data_ptr<float>(), temp.data_ptr<float>(), idx.data_ptr<int>());
  CHECK_CUDA_ERROR("furthestsampling_cuda_kernel");
  return idx;
}

REGIST_PYTORCH_EXTENSION(
    other_furthest_point_sampling, { m.def("FurthestSampling", &FurthestSampling, "FurthestSampling (CUDA)"); });
}  // namespace FarthesPointSampling