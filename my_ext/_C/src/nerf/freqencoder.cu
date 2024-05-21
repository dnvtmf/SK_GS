#include "util.cuh"

inline constexpr __device__ float PI() { return 3.141592653589793f; }

// inputs: [B, D]
// outputs: [B, C], C = D + D * deg * 2
__global__ void kernel_freq(
    const float* __restrict__ inputs, uint32_t B, uint32_t D, uint32_t deg, uint32_t C, float* outputs) {
  // parallel on per-element
  const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= B * C) return;

  // get index
  const uint32_t b = t / C;
  const uint32_t c = t - b * C;  // t % C;

  // locate
  inputs += b * D;
  outputs += t;

  // write self
  if (c < D) {
    outputs[0] = inputs[c];
    // write freq
  } else {
    const uint32_t col      = c / D - 1;
    const uint32_t d        = c % D;
    const uint32_t freq     = col / 2;
    const float phase_shift = (col % 2) * (PI() / 2);
    outputs[0]              = __sinf(scalbnf(inputs[d], freq) + phase_shift);
  }
}

// grad: [B, C], C = D + D * deg * 2
// outputs: [B, C]
// grad_inputs: [B, D]
__global__ void kernel_freq_backward(const float* __restrict__ grad, const float* __restrict__ outputs, uint32_t B,
    uint32_t D, uint32_t deg, uint32_t C, float* grad_inputs) {
  // parallel on per-element
  const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= B * D) return;

  const uint32_t b = t / D;
  const uint32_t d = t - b * D;  // t % D;

  // locate
  grad += b * C;
  outputs += b * C;
  grad_inputs += t;

  // register
  float result = grad[d];
  grad += D;
  outputs += D;

  for (uint32_t f = 0; f < deg; f++) {
    result += scalbnf(1.0f, f) * (grad[d] * outputs[D + d] - grad[D + d] * outputs[d]);
    grad += 2 * D;
    outputs += 2 * D;
  }

  // write
  grad_inputs[0] = result;
}

void freq_encode_forward(
    at::Tensor inputs, const uint32_t B, const uint32_t D, const uint32_t deg, const uint32_t C, at::Tensor outputs) {
  CHECK_CUDA(inputs);
  CHECK_CUDA(outputs);

  CHECK_CONTIGUOUS(inputs);
  CHECK_CONTIGUOUS(outputs);

  CHECK_IS_FLOATING(inputs);
  CHECK_IS_FLOATING(outputs);

  static constexpr uint32_t N_THREADS = 128;

  kernel_freq KERNEL_ARG(div_round_up(B * C, N_THREADS), N_THREADS)(
      inputs.data_ptr<float>(), B, D, deg, C, outputs.data_ptr<float>());
}

void freq_encode_backward(at::Tensor grad, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t deg,
    const uint32_t C, at::Tensor grad_inputs) {
  CHECK_CUDA(grad);
  CHECK_CUDA(outputs);
  CHECK_CUDA(grad_inputs);

  CHECK_CONTIGUOUS(grad);
  CHECK_CONTIGUOUS(outputs);
  CHECK_CONTIGUOUS(grad_inputs);

  CHECK_IS_FLOATING(grad);
  CHECK_IS_FLOATING(outputs);
  CHECK_IS_FLOATING(grad_inputs);

  static constexpr uint32_t N_THREADS = 128;

  kernel_freq_backward KERNEL_ARG(div_round_up(B * D, N_THREADS), N_THREADS)(
      grad.data_ptr<float>(), outputs.data_ptr<float>(), B, D, deg, C, grad_inputs.data_ptr<float>());
}

REGIST_PYTORCH_EXTENSION(nerf_freq_encode, {
  m.def("freq_encode_forward", &freq_encode_forward, "freq encode forward (CUDA)");
  m.def("freq_encode_backward", &freq_encode_backward, "freq encode backward (CUDA)");
})
