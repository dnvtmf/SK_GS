#include "util.cuh"

__global__ void kernel_pack_info_to_indices(
    const int* __restrict__ pack_info, const uint32_t N, const uint32_t M, int* __restrict__ indices) {
  // parallel per ray
  const uint32_t n = blockIdx.x;
  if (n >= N) return;

  uint32_t offset    = pack_info[n * 2];
  uint32_t num_steps = pack_info[n * 2 + 1];

  indices += offset * 2;
  for (int i = threadIdx.x; i < num_steps; i += blockDim.x) {
    indices[i * 2 + 0] = n;
    indices[i * 2 + 1] = i;
  }
}

Tensor pack_info_to_indices(const at::Tensor pack_info) {
  CHECK_INPUT_AND_TYPE(pack_info, torch::kInt32);
  BCNN_ASSERT(pack_info.ndimension() == 2 && pack_info.size(1) == 2, "Error shape for pack info");
  const uint32_t N = pack_info.size(0) - 1, M = pack_info[-1][0].item<int>();
  Tensor indices = at::zeros({M, 2}, pack_info.options());
  kernel_pack_info_to_indices KERNEL_ARG(N, 128)(pack_info.data_ptr<int>(), N, M, indices.data_ptr<int>());
  return indices;
}

// sigmas: [M]
// rgbs: [M, 3]
// ts: [M, 2], start, interval
// pack_info: [N, 2], offset, num_steps
// weights: [M]
// weights_sum: [N], final pixel alpha
// depth: [N,]
// image: [N, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_forward(const scalar_t* __restrict__ sigmas,
    const scalar_t* __restrict__ ts, const int* __restrict__ pack_info, const scalar_t* __restrict__ rgbs,
    const uint32_t M, const uint32_t N, const float T_thresh, bool is_alpha, scalar_t* __restrict__ weights,
    scalar_t* __restrict__ weights_sum, scalar_t* __restrict__ depth, scalar_t* __restrict__ image) {
  // parallel per ray
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  // locate
  uint32_t offset    = pack_info ? pack_info[n * 2] : n * (M / N);
  uint32_t num_steps = pack_info ? pack_info[n * 2 + 1] : (M / N);

  // empty ray, or ray that exceed max step count.
  if (num_steps == 0 || offset + num_steps > M) {
    weights_sum[n] = 0;
    if (depth != nullptr) depth[n] = 0;
    if (image != nullptr) {
      image[n * 3]     = 0;
      image[n * 3 + 1] = 0;
      image[n * 3 + 2] = 0;
    }
    return;
  }

  ts += offset * 2;
  if (ts[0] < 0) return;  // invalid value

  weights += offset;
  sigmas += offset;
  if (rgbs != nullptr) rgbs += offset * 3;

  // accumulate
  float T = 1.0f;
  float r = 0, g = 0, b = 0, d = 0;

  for (uint32_t step = 0; step < num_steps; step++) {
    const float alpha  = clamp<float>(is_alpha ? (float) sigmas[0] : 1.0f - __expf(-sigmas[0] * ts[1]));
    const float weight = alpha * T;

    weights[0] = weight;

    if (rgbs != nullptr) {
      r += weight * rgbs[0];
      g += weight * rgbs[1];
      b += weight * rgbs[2];
    }
    // ws += weight;
    d += weight * ts[0];

    T *= 1.0f - alpha;

    // minimal remained transmittence
    if (T < T_thresh) break;

    // printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

    // locate
    weights++;
    sigmas++;
    ts += 2;
    if (rgbs != nullptr) rgbs += 3;
  }

  // printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

  // write
  weights_sum[n] = 1.f - T;  // weights_sum
  if (depth != nullptr) depth[n] = d;
  if (rgbs != nullptr) {
    image[n * 3]     = r;
    image[n * 3 + 1] = g;
    image[n * 3 + 2] = b;
  }
}

std::tuple<Tensor, Tensor, at::optional<Tensor>, at::optional<Tensor>> composite_rays_forward(const at::Tensor sigmas,
    const at::Tensor ts, const at::optional<at::Tensor> pack_info, const at::optional<at::Tensor> rgbs,
    const float T_thresh, bool need_depth, bool is_alpha) {
  CHECK_INPUT(sigmas);

  if (pack_info.has_value()) {
    CHECK_INPUT(pack_info.value());
    BCNN_ASSERT(pack_info.value().ndimension() == 2 && pack_info.value().size(1) == 2, "Error shape for pack_info");
    BCNN_ASSERT(sigmas.ndimension() == 1, "Error shape for simgas");
  } else {
    BCNN_ASSERT(sigmas.ndimension() == 2, "Error shape for simgas");
  }
  const uint32_t M = sigmas.numel();
  const uint32_t N = pack_info.has_value() ? pack_info.value().size(0) - 1 : sigmas.size(0);

  if (rgbs.has_value()) {
    CHECK_INPUT(rgbs.value());
    BCNN_ASSERT(rgbs.value().numel() == M * 3, "Error shape for rgbs");
  }

  Tensor weights     = at::zeros_like(sigmas);
  Tensor weights_sum = at::zeros({N}, sigmas.options());
  at::optional<Tensor> depth, image;
  if (need_depth) depth = at::zeros({N}, sigmas.options());
  if (rgbs.has_value()) image = at::zeros({N, 3}, rgbs.value().options());

  static constexpr uint32_t N_THREAD = 128;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.scalar_type(), "composite_rays_forward", ([&] {
    kernel_composite_rays_train_forward KERNEL_ARG(div_round_up(N, N_THREAD), N_THREAD)(sigmas.data_ptr<scalar_t>(),
        ts.data_ptr<scalar_t>(), pack_info.has_value() ? pack_info.value().data_ptr<int>() : nullptr,
        rgbs.has_value() ? rgbs.value().data_ptr<scalar_t>() : nullptr, M, N, T_thresh, is_alpha,
        weights.data_ptr<scalar_t>(), weights_sum.data_ptr<scalar_t>(),
        depth.has_value() ? depth.value().data_ptr<scalar_t>() : nullptr,
        image.has_value() ? image.value().data_ptr<scalar_t>() : nullptr);
  }));
  return std::make_tuple(weights, weights_sum, image, depth);
}

// M: total number of samples
// N: the number of rays
// grad_weights: [M,]
// grad_weights_sum: [N,]
// grad_image: [N, 3]
// grad_depth: [N,]
// sigmas: [M]
// rgbs: [M, 3]
// ts: [M, 2], start interval
// pack_info: [N, 2], offset, num_steps
// weights_sum: [N,], weights_sum here
// image: [N, 3]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_backward(const scalar_t* __restrict__ grad_weights,
    const scalar_t* __restrict__ grad_weights_sum, const scalar_t* __restrict__ grad_depth,
    const scalar_t* __restrict__ grad_image, const scalar_t* __restrict__ sigmas, const scalar_t* __restrict__ rgbs,
    const scalar_t* __restrict__ ts, const int* __restrict__ pack_info, const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ weights_sum, const scalar_t* __restrict__ depth, const scalar_t* __restrict__ image,
    const uint32_t M, const uint32_t N, const float T_thresh, bool is_alpha, scalar_t* __restrict__ grad_sigmas,
    scalar_t* __restrict__ grad_rgbs) {
  // parallel per ray
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  // locate
  uint32_t offset    = pack_info ? pack_info[n * 2] : n * (M / N);
  uint32_t num_steps = pack_info ? pack_info[n * 2 + 1] : M / N;

  if (num_steps == 0 || offset + num_steps > M) return;

  grad_weights += offset;
  weights += offset;
  sigmas += offset;
  grad_sigmas += offset;
  ts += offset * 2;
  if (ts[0] < 0) return;

  weights_sum += n;
  // grad_weights_sum += n;
  const float grad_ws = grad_weights_sum[n];
  if (grad_depth) {
    grad_depth += n;
    depth += n;
  }
  if (grad_image) {
    grad_image += n * 3;
    image += n * 3;
    rgbs += offset * 3;
  }
  if (grad_rgbs) grad_rgbs += offset * 3;

  // accumulate
  float T       = 1.0f;
  float r_final = 0, g_final = 0, b_final = 0, d_final = grad_depth ? (float) depth[0] : 0, ws_final = 0;
  float r = 0, g = 0, b = 0, ws = 0, d = 0;
  if (grad_image) r_final = image[0], g_final = image[1], b_final = image[2];

  for (uint32_t step = 0; step < num_steps; ++step) ws_final += (grad_weights[step] + grad_ws) * weights[step];

  for (uint32_t step = 0; step < num_steps; step++) {
    const float alpha  = clamp<float>(is_alpha ? (float) sigmas[0] : 1.0f - __expf(-sigmas[0] * ts[1]));
    const float weight = alpha * T;

    if (grad_image) {
      r += weight * rgbs[0];
      g += weight * rgbs[1];
      b += weight * rgbs[2];
    }
    ws += weight * (grad_weights[step] + grad_ws);
    if (grad_depth) d += weight * ts[0];

    T *= 1.0f - alpha;

    float _grad_simga = (grad_ws + grad_weights[step]) * T - (ws_final - ws);
    // write grad_rgbs
    if (grad_rgbs) {
      grad_rgbs[0] = grad_image[0] * weight;
      grad_rgbs[1] = grad_image[1] * weight;
      grad_rgbs[2] = grad_image[2] * weight;
    }
    if (grad_image) {
      _grad_simga += grad_image[0] * (T * rgbs[0] - (r_final - r));
      _grad_simga += grad_image[1] * (T * rgbs[1] - (g_final - g));
      _grad_simga += grad_image[2] * (T * rgbs[2] - (b_final - b));
    }

    // write grad_sigmas
    if (grad_depth) _grad_simga += grad_depth[0] * (T * ts[0] - (d_final - d));
    grad_sigmas[0] = (is_alpha ? 1.0f / clamp(1.0f - alpha, 1e-6f) : (float) ts[1]) * _grad_simga;

    //  minimal remained transmittence
    if (T < T_thresh) break;

    // locate
    sigmas++;
    if (grad_image) rgbs += 3;
    if (grad_rgbs) grad_rgbs += 3;
    ts += 2;
    grad_sigmas++;
  }
}

void composite_rays_backward(const Tensor grad_weights, Tensor grad_weights_sum, const at::optional<Tensor> grad_depth,
    const at::optional<Tensor> grad_image, const at::Tensor sigmas, const at::optional<Tensor> rgbs,
    const at::Tensor ts, const at::optional<Tensor> pack_info, const at::Tensor weights, const at::Tensor weights_sum,
    const at::optional<Tensor> depth, at::optional<Tensor> image, const uint32_t M, const uint32_t N,
    const float T_thresh, bool is_alpha, at::Tensor grad_sigmas, at::optional<Tensor> grad_rgbs) {
  static constexpr uint32_t N_THREAD = 128;
  CHECK_INPUT(grad_weights);
  CHECK_INPUT(grad_weights_sum);
  CHECK_INPUT(sigmas);
  CHECK_INPUT(ts);
  CHECK_INPUT(weights);
  CHECK_INPUT(weights_sum);
  CHECK_INPUT(grad_sigmas);
  if (grad_depth.has_value()) CHECK_INPUT(grad_depth.value());
  if (grad_image.has_value()) CHECK_INPUT(grad_image.value());
  if (rgbs.has_value()) CHECK_INPUT(rgbs.value());
  if (depth.has_value()) CHECK_INPUT(depth.value());
  if (image.has_value()) CHECK_INPUT(image.value());
  if (grad_rgbs.has_value()) CHECK_INPUT(grad_rgbs.value());
  if (pack_info.has_value()) CHECK_INPUT_AND_TYPE(pack_info.value(), torch::kInt32);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.scalar_type(), "composite_rays_backward", ([&] {
    kernel_composite_rays_train_backward<scalar_t> KERNEL_ARG(div_round_up(N, N_THREAD), N_THREAD)(
        grad_weights.data_ptr<scalar_t>(), grad_weights_sum.data_ptr<scalar_t>(),
        grad_depth.has_value() ? grad_depth.value().data_ptr<scalar_t>() : nullptr,
        grad_image.has_value() ? grad_image.value().data_ptr<scalar_t>() : nullptr, sigmas.data_ptr<scalar_t>(),
        rgbs.has_value() ? rgbs.value().data_ptr<scalar_t>() : nullptr, ts.data_ptr<scalar_t>(),
        pack_info.has_value() ? pack_info.value().data_ptr<int>() : nullptr, weights.data_ptr<scalar_t>(),
        weights_sum.data_ptr<scalar_t>(), depth.has_value() ? depth.value().data_ptr<scalar_t>() : nullptr,
        image.has_value() ? image.value().data_ptr<scalar_t>() : nullptr, M, N, T_thresh, is_alpha,
        grad_sigmas.data_ptr<scalar_t>(), grad_rgbs.has_value() ? grad_rgbs.value().data_ptr<scalar_t>() : nullptr);
  }));
}

template <typename scalar_t>
__global__ void kernel_composite_rays_step(const uint32_t n_alive, const uint32_t n_step, const float T_thresh,
    int* rays_alive, scalar_t* rays_t, const scalar_t* __restrict__ sigmas, const scalar_t* __restrict__ rgbs,
    const scalar_t* __restrict__ ts, scalar_t* weights_sum, scalar_t* depth, scalar_t* image) {
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= n_alive) return;

  const int index = rays_alive[n];  // ray id

  // locate
  sigmas += n * n_step;
  rgbs += n * n_step * 3;
  ts += n * n_step * 2;

  rays_t += index;
  weights_sum += index;
  depth += index;
  image += index * 3;

  float t;
  float d = depth[0], r = image[0], g = image[1], b = image[2], weight_sum = weights_sum[0];

  // accumulate
  uint32_t step = 0;
  while (step < n_step) {
    // ray is terminated if t == 0
    if (ts[0] == 0) break;

    const float alpha = clamp<float>(1.0f - __expf(-sigmas[0] * ts[1]));

    /*
    T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
    w_i = alpha_i * T_i
    -->
    T_i = 1 - \sum_{j=0}^{i-1} w_j
    */
    const float T      = 1 - weight_sum;
    const float weight = alpha * T;
    weight_sum += weight;

    t = ts[0];
    d += weight * t;  // real depth
    r += weight * rgbs[0];
    g += weight * rgbs[1];
    b += weight * rgbs[2];

    // printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

    // ray is terminated if T is too small
    // use a larger bound to further accelerate inference
    if (T < T_thresh) break;

    // locate
    sigmas++;
    rgbs += 3;
    ts += 2;
    step++;
  }

  // printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

  // rays_alive = -1 means ray is terminated early.
  if (step < n_step) {
    rays_alive[n] = -1;
  } else {
    rays_t[0] = t;
  }

  weights_sum[0] = weight_sum;  // this is the thing I needed!
  depth[0]       = d;
  image[0]       = r;
  image[1]       = g;
  image[2]       = b;
}

void composite_packed_rays_step(const uint32_t n_alive, const uint32_t n_step, const float T_thresh,
    at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor ts, at::Tensor weights,
    at::Tensor depth, at::Tensor image) {
  static constexpr uint32_t N_THREAD = 128;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(image.scalar_type(), "composite_packed_rays_step", ([&] {
    kernel_composite_rays_step KERNEL_ARG(div_round_up(n_alive, N_THREAD), N_THREAD)(n_alive, n_step, T_thresh,
        rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(),
        ts.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
  }));
}

REGIST_PYTORCH_EXTENSION(nerf_volume_rendering, {
  m.def("pack_info_to_indices", &pack_info_to_indices, "pack_info_to_indices (CUDA)");
  m.def("composite_rays_forward", &composite_rays_forward, "composite_rays_forward (CUDA)");
  m.def("composite_rays_backward", &composite_rays_backward, "composite_rays_backward (CUDA)");
  m.def("composite_packed_rays_step", &composite_packed_rays_step, "composite rays (CUDA)");
});