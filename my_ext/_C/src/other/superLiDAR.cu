#include "util.cuh"
namespace MySuperPoint {

template <typename T>
__global__ void superpoint_loss_foward(int N, int M, int K, int F, T alpha, T beta, const T *__restrict__ feat_p,
    const T *__restrict__ feat_sp, const int *__restrict__ neighbor, const int *__restrict__ p2sp,
    T *__restrict__ loss) {
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;
  extern __shared__ int smem[];
  int *Ni  = smem;
  int *Ti  = Ni + blockDim.x * K;
  int *Nsp = Ti + blockDim.x * K;
  int nT   = 0;

  Ni += threadIdx.x * K;
  Ti += threadIdx.x * K;
  Nsp += threadIdx.x * K;
  neighbor += n * K;

  for (int i = 0; i < K; ++i) {
    Ni[i]      = neighbor[i];
    bool in_Ti = false;
    Nsp[i]     = p2sp[Ni[i]];
    for (int j = 0; j < nT; ++j) {
      if (Nsp[i] == Ti[j]) {
        in_Ti = true;
        break;
      }
    }
    if (!in_Ti) Ti[nT++] = Nsp[i];
  }

  T loss_n = 0;
  for (int i = 0; i < K; ++i) {
    int j = Ni[i];
    for (int k = 0; k < nT; ++k) {
      T dist = 0, scale;
      for (int f = 0; f < F; ++f) {
        T df = feat_p[j * F + f] - feat_sp[Ti[k] * F + f];
        dist += df * df;
      }
      if (Nsp[i] == Ti[k]) {
        dist  = dist - alpha;
        scale = T(1) / K;
      } else {
        dist  = beta - dist;
        scale = T(1) / T((nT - 1) * K);
      }
      loss_n += dist > 0 ? dist * dist * scale : 0;
    }
  }
  __syncthreads();
  reduce_sum_block<T, false>(loss_n);
  if (threadIdx.x == 0) atomicAdd(loss, loss_n / N);
}

template <typename T>
__global__ void superpoint_loss_backward_kernel(int N, int M, int K, int F, T alpha, T beta,
    const T *__restrict__ feat_p, const T *__restrict__ feat_sp, const int *__restrict__ neighbor,
    const int *__restrict__ p2sp, const T grad_loss, T *__restrict__ grad_feat_p, T *__restrict__ grad_feat_sp) {
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;
  extern __shared__ int smem[];
  int *Ni  = smem;
  int *Ti  = Ni + blockDim.x * K;
  int *Nsp = Ti + blockDim.x * K;
  T *dfp   = (T *) &Nsp[blockDim.x * K];
  int nT   = 0;

  Ni += threadIdx.x * K;
  Ti += threadIdx.x * K;
  Nsp += threadIdx.x * K;
  dfp += threadIdx.x * F;
  neighbor += n * K;

  for (int i = 0; i < K; ++i) {
    Ni[i]      = neighbor[i];
    bool in_Ti = false;
    Nsp[i]     = p2sp[Ni[i]];
    for (int j = 0; j < nT; ++j) {
      if (Nsp[i] == Ti[j]) {
        in_Ti = true;
        break;
      }
    }
    if (!in_Ti) Ti[nT++] = Nsp[i];
  }

  for (int k = 0; k < nT; ++k) {
    for (int f = 0; f < F; ++f) dfp[f] = 0;

    for (int i = 0; i < K; ++i) {
      int j  = Ni[i];
      T dist = 0;
      for (int f = 0; f < F; ++f) {
        T df = feat_p[j * F + f] - feat_sp[Ti[k] * F + f];
        dist += df * df;
      }
      T scale;
      if (Nsp[i] == Ti[k]) {
        dist  = dist - alpha;
        scale = T(4) / K;
      } else {
        dist  = beta - dist;
        scale = T(-4) / (K * (nT - 1));
      }
      if (dist > 0) {
        dist = scale * dist * grad_loss;
        for (int f = 0; f < F; ++f) {
          T grad = dist * (feat_p[j * F + f] - feat_sp[Ti[k] * F + f]);
          atomicAdd(grad_feat_p + j * F + f, grad);
          //   atomicAdd(grad_feat_sp + Ti[k] * F + f, -temp * df);
          dfp[f] -= grad;
        }
      }
    }
    for (int f = 0; f < F; ++f) atomicAdd(grad_feat_sp + Ti[k] * F + f, dfp[f]);
  }
}

Tensor superpoint_loss_forward(
    Tensor &feat_p, Tensor &feat_sp, Tensor &neighbor, Tensor &p2sp, float alpha, float beta) {
  CHECK_INPUT(feat_p);
  CHECK_INPUT(feat_sp);
  BCNN_ASSERT(feat_p.scalar_type() == feat_sp.scalar_type(), "dtype for feat_p and feat_sp must be same");
  CHECK_INPUT_AND_TYPE(neighbor, torch::kInt32);
  CHECK_INPUT_AND_TYPE(p2sp, torch::kInt32);
  CHECK_NDIM(feat_p, 2);
  int N = feat_p.size(0);
  int F = feat_p.size(1);
  BCNN_ASSERT(feat_sp.ndimension() == 2 && feat_sp.size(1) == F, "Error shape for feat_sp");
  int M = feat_sp.size(0);
  BCNN_ASSERT(neighbor.ndimension() == 2 && neighbor.size(0) == N, "Error shaep for neighbor");
  int K = neighbor.size(1);
  CHECK_SHAPE(p2sp, N);
  Tensor loss = torch::zeros({1}, feat_p.options());
  AT_DISPATCH_FLOATING_TYPES(feat_p.scalar_type(), "superpoint_loss_foward", [&] {
    const int num_threads = 256;
    superpoint_loss_foward KERNEL_ARG(div_round_up(N, num_threads), num_threads, num_threads * K * (sizeof(int) * 3))(N,
        M, K, F, (scalar_t) alpha, (scalar_t) beta, feat_p.contiguous().data_ptr<scalar_t>(),
        feat_sp.data_ptr<scalar_t>(), neighbor.contiguous().data_ptr<int>(), p2sp.contiguous().data_ptr<int>(),
        loss.data_ptr<scalar_t>());
    CHECK_CUDA_ERROR("superpoint_loss_foward");
  });
  return loss;
}

std::tuple<Tensor, Tensor> superpoint_loss_backward(
    Tensor grad_loss, Tensor &feat_p, Tensor &feat_sp, Tensor &neighbor, Tensor &p2sp, float alpha, float beta) {
  int N                 = feat_p.size(0);
  int F                 = feat_p.size(1);
  int M                 = feat_sp.size(0);
  int K                 = neighbor.size(1);
  Tensor grad_feat_p    = torch::zeros_like(feat_p);
  Tensor grad_feat_sp   = torch::zeros_like(feat_sp);
  const int num_threads = 256;
  AT_DISPATCH_FLOATING_TYPES(feat_p.scalar_type(), "superpoint_loss_backward_kernel", [&] {
    superpoint_loss_backward_kernel KERNEL_ARG(
        div_round_up(N, num_threads), num_threads, num_threads * (K * sizeof(int) * 3 + F * sizeof(scalar_t)))(N, M, K,
        F, (scalar_t) alpha, (scalar_t) beta, feat_p.contiguous().data_ptr<scalar_t>(), feat_sp.data_ptr<scalar_t>(),
        neighbor.contiguous().data_ptr<int>(), p2sp.contiguous().data_ptr<int>(), grad_loss.item<scalar_t>() / N,
        grad_feat_p.data_ptr<scalar_t>(), grad_feat_sp.data_ptr<scalar_t>());
    CHECK_CUDA_ERROR("superpoint_loss_backward_kernel");
  });

  return std::make_tuple(grad_feat_p, grad_feat_sp);
}

void SuperLiDARBFS(Tensor &points, Tensor &embed, double gamma, double beta, int N_min, int N_max) {
  // LiDAR Point Grouping
}

REGIST_PYTORCH_EXTENSION(other_my_superpoint, {
  m.def("superpoint_loss_forward", &superpoint_loss_forward, "superpoint_loss_forward (CUDA)");
  m.def("superpoint_loss_backward", &superpoint_loss_backward, "superpoint_loss_backward (CUDA)");
});
}  // namespace MySuperPoint