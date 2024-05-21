#include "util.cuh"

// just for compatability of half precision in AT_DISPATCH_FLOATING_TYPES_AND_HALF... program will never reach here!
__device__ inline at::Half atomicAdd(at::Half *address, at::Half val) {
  // requires CUDA >= 10 and ARCH >= 70
  // this is very slow compared to float or __half2, never use it.
  // return atomicAdd(reinterpret_cast<__half*>(address), val);
}

template <typename T>
__device__ inline T smoothstep(T val) {
  return val * val * (3.0f - 2.0f * val);
}

template <typename T>
__device__ inline T smoothstep_derivative(T val) {
  return 6.0f * val * (1.0f - val);
}

template <typename T>
__device__ inline T smoothstep_derivative_derivative(T val) {
  return 6.0f - 12.0f * val;
}

template <uint32_t D>
__device__ uint32_t fast_hash(const uint32_t pos_grid[D]) {
  // static_assert(D <= 7, "fast_hash can only hash up to 7 dimensions.");

  // While 1 is technically not a good prime for hashing (or a prime at all), it helps memory coherence
  // and is sufficient for our use case of obtaining a uniformly colliding index from high-dimensional
  // coordinates.
  // coherent type of hashing
  constexpr uint32_t primes[7] = {1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u};

  uint32_t result = 0;
#pragma unroll
  for (uint32_t i = 0; i < D; ++i) {
    result ^= pos_grid[i] * primes[i];
  }

  return result;
}

template <uint32_t D, uint32_t C>
__device__ uint32_t get_grid_index(const uint32_t gridtype, const bool align_corners, const uint32_t ch,
    const uint32_t hashmap_size, const uint32_t resolution, const uint32_t pos_grid[D]) {
  uint32_t stride = 1;
  uint32_t index  = 0;

#pragma unroll
  for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
    index += pos_grid[d] * stride;
    stride *= align_corners ? resolution : (resolution + 1);
  }

  // NOTE: for NeRF, the hash is in fact not necessary. Check https://github.com/NVlabs/instant-ngp/issues/97.
  // gridtype: 0 == hash, 1 == tiled
  if (gridtype == 0 && stride > hashmap_size) {
    index = fast_hash<D>(pos_grid);
  }

  return (index % hashmap_size) * C + ch;
}

template <uint32_t D>
__device__ uint32_t get_grid_index(const uint32_t pos_grid[D]) {
  uint32_t stride = 1;
  uint32_t index  = 0;

#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    index += pos_grid[d] * stride;
    stride *= 2;
  }
  return index;
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid_encode_forward(const float *__restrict__ inputs, const scalar_t *__restrict__ grid,
    const int *__restrict__ offsets, scalar_t *__restrict__ outputs, const uint32_t B, const uint32_t L, const float S,
    const uint32_t H, scalar_t *__restrict__ dy_dx, const uint32_t valid_level, const uint32_t gridtype,
    const bool align_corners, const uint32_t interp) {
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= B) return;

  const uint32_t level = blockIdx.y;

  // locate
  grid += (uint32_t) offsets[level] * C;
  inputs += b * D;
  outputs += level * B * C + b * C;

  // check input range (should be in [0, 1]) and level <= valid level
  bool flag_oob = level > valid_level;
#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    if (inputs[d] < 0 || inputs[d] > 1) {
      flag_oob = true;
    }
  }
  // if input out of bound, just set output to 0
  if (flag_oob) {
#pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
      outputs[ch] = 0;
    }
    if (dy_dx) {
      dy_dx += b * D * L * C + level * D * C;  // B L D C
#pragma unroll
      for (uint32_t d = 0; d < D; d++) {
#pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
          dy_dx[d * C + ch] = 0;
        }
      }
    }
    return;
  }

  const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
  const float scale           = exp2f(level * S) * H - 1.0f;
  const uint32_t resolution   = (uint32_t) ceil(scale) + 1;

  // calculate coordinate (always use float for precision!)
  float pos[D];
  float pos_deriv[D];  // linear deriv is default to 1, bug pos_deriv[1:] = 0
  uint32_t pos_grid[D];

#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    pos[d]      = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
    pos_grid[d] = floorf(pos[d]);
    pos[d] -= (float) pos_grid[d];
    // smoothstep instead of linear
    if (interp == 1) {
      pos_deriv[d] = smoothstep_derivative(pos[d]);
      pos[d]       = smoothstep(pos[d]);
    } else {
      pos_deriv[d] = 1.f;
    }
  }

  // interpolate
  scalar_t results[C] = {0};  // temp results in register

#pragma unroll
  for (uint32_t idx = 0; idx < (1 << D); idx++) {
    float w = 1;
    uint32_t pos_grid_local[D];

#pragma unroll
    for (uint32_t d = 0; d < D; d++) {
      if ((idx & (1 << d)) == 0) {
        w *= 1 - pos[d];
        pos_grid_local[d] = pos_grid[d];
      } else {
        w *= pos[d];
        pos_grid_local[d] = pos_grid[d] + 1;
      }
    }

    uint32_t index = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

// writing to register (fast)
#pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
      results[ch] += w * grid[index + ch];
    }
  }

// writing to global memory (slow)
#pragma unroll
  for (uint32_t ch = 0; ch < C; ch++) {
    outputs[ch] = results[ch];
  }

  // prepare dy_dx
  // differentiable (soft) indexing: https://discuss.pytorch.org/t/differentiable-indexing/17647/9
  if (dy_dx) {
    dy_dx += b * L * D * C + level * D * C;  // B L D C

#pragma unroll
    for (uint32_t gd = 0; gd < D; gd++) {
      scalar_t results_grad[C] = {0};

#pragma unroll
      for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
        float w = scale;
        uint32_t pos_grid_local[D];

#pragma unroll
        for (uint32_t nd = 0; nd < D - 1; nd++) {
          const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

          if ((idx & (1 << nd)) == 0) {
            w *= 1 - pos[d];
            pos_grid_local[d] = pos_grid[d];
          } else {
            w *= pos[d];
            pos_grid_local[d] = pos_grid[d] + 1;
          }
        }

        pos_grid_local[gd] = pos_grid[gd];
        uint32_t index_left =
            get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);
        pos_grid_local[gd] = pos_grid[gd] + 1;
        uint32_t index_right =
            get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

#pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
          results_grad[ch] += w * (grid[index_right + ch] - grid[index_left + ch]) * pos_deriv[gd];
        }
      }
#pragma unroll
      for (uint32_t ch = 0; ch < C; ch++) {
        dy_dx[gd * C + ch] = results_grad[ch];
      }
    }
  }
}

template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C = 2>
__global__ void kernel_grid_backward(const scalar_t *__restrict__ grad, const float *__restrict__ inputs,
    const scalar_t *__restrict__ grid, const int *__restrict__ offsets, scalar_t *__restrict__ grad_grid,
    const uint32_t B, const uint32_t L, const float S, const uint32_t H, const uint32_t valid_level,
    const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
  const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
  if (b >= B) return;

  const uint32_t level = blockIdx.y;
  if (level > valid_level) return;

  const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;
  // locate
  grad_grid += offsets[level] * C;
  inputs += b * D;
  grad += level * B * C + b * C + ch;  // L, B, C

  const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
  const float scale           = exp2f(level * S) * H - 1.0f;
  const uint32_t resolution   = (uint32_t) ceil(scale) + 1;

// check input range (should be in [0, 1])
#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    if (inputs[d] < 0 || inputs[d] > 1) {
      return;  // grad is init as 0, so we simply return.
    }
  }

  // calculate coordinate
  float pos[D];
  uint32_t pos_grid[D];

#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    pos[d]      = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
    pos_grid[d] = floorf(pos[d]);
    pos[d] -= (float) pos_grid[d];
    // smoothstep instead of linear
    if (interp == 1) {
      pos[d] = smoothstep(pos[d]);
    }
  }

  scalar_t grad_cur[N_C] = {0};  // fetch to register
#pragma unroll
  for (uint32_t c = 0; c < N_C; c++) {
    grad_cur[c] = grad[c];
  }

// interpolate
#pragma unroll
  for (uint32_t idx = 0; idx < (1 << D); idx++) {
    float w = 1;
    uint32_t pos_grid_local[D];

#pragma unroll
    for (uint32_t d = 0; d < D; d++) {
      if ((idx & (1 << d)) == 0) {
        w *= 1 - pos[d];
        pos_grid_local[d] = pos_grid[d];
      } else {
        w *= pos[d];
        pos_grid_local[d] = pos_grid[d] + 1;
      }
    }

    uint32_t index = get_grid_index<D, C>(gridtype, align_corners, ch, hashmap_size, resolution, pos_grid_local);

    // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
    // use float which is better than __half, if N_C % 2 != 0
    if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
#pragma unroll
      for (uint32_t c = 0; c < N_C; c += 2) {
        // process two __half at once (by interpreting as a __half2)
        __half2 v = {(__half) (w * grad_cur[c]), (__half) (w * grad_cur[c + 1])};
        atomicAdd((__half2 *) &grad_grid[index + c], v);
      }
      // float, or __half when N_C % 2 != 0 (which means C == 1)
    } else {
#pragma unroll
      for (uint32_t c = 0; c < N_C; c++) {
        atomicAdd(&grad_grid[index + c], w * grad_cur[c]);
      }
    }
  }
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward(const scalar_t *__restrict__ dL_dy, const scalar_t *__restrict__ dy_dx,
    scalar_t *__restrict__ dL_dx, uint32_t B, uint32_t L) {
  const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= B * D) return;

  const uint32_t b = t / D;
  const uint32_t d = t - b * D;

  dy_dx += b * L * D * C;

  scalar_t result = 0;

#pragma unroll
  for (int l = 0; l < L; l++) {
#pragma unroll
    for (int ch = 0; ch < C; ch++) {
      result += dL_dy[l * B * C + b * C + ch] * dy_dx[l * D * C + d * C + ch];
    }
  }

  dL_dx[t] = result;
}

template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C = 2>
__global__ void kernel_second_backward_grad(const scalar_t *__restrict__ grad, const float *__restrict__ inputs,
    const scalar_t *__restrict__ grid, const int *__restrict__ offsets, const float *__restrict__ grad_grad_inputs,
    const scalar_t *__restrict__ dy_dx, scalar_t *__restrict__ grad_grad, const uint32_t B, const uint32_t L,
    const float S, const uint32_t H, const uint32_t valid_level, const uint32_t gridtype, const bool align_corners,
    const uint32_t interp) {
  // grad_grad_inputs: [B, D], float
  // grad_grad: [L, B, C], float
  // grad2_embedding: [s0, C], float
  // grad: [L, B, C], float
  // inputs: [B, D], float, in [0, 1]
  // embeddings: [sO, C], float
  // offsets: [L + 1], uint32_t
  // dy_dx: [B, L * D * C]
  // H: base resolution

  const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
  if (b >= B) return;

  const uint32_t level = blockIdx.y;
  if (level > valid_level) return;
  const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

  // locate
  grad_grad += level * B * C + b * C + ch;
  grad += level * B * C + b * C + ch;  // L, B, C

  grad_grad_inputs += b * D;
  dy_dx += b * L * D * C + level * D * C + ch;

  scalar_t result[N_C] = {0};

#pragma unroll
  for (int d = 0; d < D; d++) {
#pragma unroll
    for (int c = 0; c < N_C; c++) {
      result[c] += grad_grad_inputs[d] * dy_dx[d * C + c];
    }
  }

  // write to global memory
  for (int c = 0; c < N_C; c++) {
    grad_grad[c] = result[c];
  }

  // printf("[b=%d, l=%d, ch=%d, D=%d, C=%d, N_C=%d, grad_grad=(%f, %f)]\n", b, level, ch, D, C, N_C, result[0],
  // result[1]);
}

template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C = 2>
__global__ void kernel_second_backward_embedding(const scalar_t *__restrict__ grad, const float *__restrict__ inputs,
    const scalar_t *__restrict__ grid, const int *__restrict__ offsets, const float *__restrict__ grad_grad_inputs,
    scalar_t *__restrict__ grad2_embeddings, const uint32_t B, const uint32_t L, const float S, const uint32_t H,
    const int valid_level, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
  const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
  if (b >= B) return;

  const uint32_t level = blockIdx.y;
  if (level > valid_level) return;
  const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

  // grad_grad_inputs: [B, D], float
  // grad_grad: [L, B, C], float
  // grad2_embedding: [s0, C], float
  // grad: [L, B, C], float
  // inputs: [B, D], float, in [0, 1]
  // embeddings: [sO, C], float
  // offsets: [L + 1], uint32_t
  // H: base resolution

  // locate
  grad2_embeddings += offsets[level] * C;
  inputs += b * D;
  grad += level * B * C + b * C + ch;  // L, B, C
  grad_grad_inputs += b * D;

  const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
  const float scale           = exp2f(level * S) * H - 1.0f;
  const uint32_t resolution   = (uint32_t) ceil(scale) + 1;

// check input range (should be in [0, 1])
#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    if (inputs[d] < 0 || inputs[d] > 1) {
      return;  // grad is init as 0, so we simply return.
    }
  }

  // calculate coordinate
  float pos[D];
  float pos_derivative[D];
  uint32_t pos_grid[D];

#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    pos[d]      = (float) inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
    pos_grid[d] = floorf(pos[d]);
    pos[d] -= (float) pos_grid[d];
    if (interp == 1) {
      pos_derivative[d] = smoothstep_derivative(pos[d]);
      pos[d]            = smoothstep(pos[d]);
    } else {
      pos_derivative[d] = 1.0f;
    }
  }

  scalar_t grad_cur[N_C] = {0};  // fetch to register
#pragma unroll
  for (uint32_t c = 0; c < N_C; c++) {
    grad_cur[c] = grad[c];
  }

  scalar_t grad2_input_cur[D] = {0};  // fetch to register
  for (uint32_t d = 0; d < D; d++) {
    grad2_input_cur[d] = grad_grad_inputs[d];
  }

  // grad2_embeddings cache
  // don't need N_C
  scalar_t grad_embeddings_cur[N_C * (1 << D)] = {0};  // cache [2^D * N_C]

// compute gradients
#pragma unroll
  for (uint32_t gd = 0; gd < D; gd++) {
    scalar_t results_grad[C] = {0};

#pragma unroll
    for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
      float w = scale;
      uint32_t pos_grid_local[D];

#pragma unroll
      for (uint32_t nd = 0; nd < D - 1; nd++) {
        const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

        if ((idx & (1 << nd)) == 0) {
          w *= 1 - pos[d];
          pos_grid_local[d] = 0;
        } else {
          w *= pos[d];
          pos_grid_local[d] = 1;
        }
      }

      pos_grid_local[gd]  = 0;
      uint32_t index_left = get_grid_index<D>(pos_grid_local);
      // uint32_t index_left = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution,
      // pos_grid_local);
      pos_grid_local[gd]   = 1;
      uint32_t index_right = get_grid_index<D>(pos_grid_local);
      // uint32_t index_right = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution,
      // pos_grid_local);

#pragma unroll
      for (uint32_t c = 0; c < N_C; c++) {
        grad_embeddings_cur[index_right * N_C + c] += w * grad_cur[c] * grad2_input_cur[gd] * pos_derivative[gd];
        grad_embeddings_cur[index_left * N_C + c] -= w * grad_cur[c] * grad2_input_cur[gd] * pos_derivative[gd];
      }
    }
  }

// write to global memory
#pragma unroll
  for (uint32_t idx = 0; idx < (1 << D); idx++) {
    uint32_t pos_grid_local[D];
    uint32_t cache_index = 0;
    uint32_t stride      = 1;

#pragma unroll
    for (uint32_t d = 0; d < D; d++) {
      if ((idx & (1 << d)) == 0) {
        pos_grid_local[d] = pos_grid[d];
      } else {
        pos_grid_local[d] = pos_grid[d] + 1;
        cache_index += stride;
      }
      stride *= 2;
    }
    // uint32_t index = get_grid_index<D, C>(ch, hashmap_size, resolution, pos_grid_local);
    uint32_t index = get_grid_index<D, C>(gridtype, align_corners, ch, hashmap_size, resolution, pos_grid_local);

    // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
    // TODO: use float which is better than __half, if N_C % 2 != 0
    if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
#pragma unroll
      for (uint32_t c = 0; c < N_C; c += 2) {
        // process two __half at once (by interpreting as a __half2)
        __half2 v = {(__half) (1.0 * grad_embeddings_cur[cache_index * N_C + c]),
            (__half) (1.0 * grad_embeddings_cur[cache_index * N_C + c + 1])};
        //__half2 v ={0, 0};
        atomicAdd((__half2 *) &grad2_embeddings[index + c], v);
      }
      // float, or __half when N_C % 2 != 0 (which means C == 1)
    } else {
#pragma unroll
      for (uint32_t c = 0; c < N_C; c++) {
        atomicAdd(&grad2_embeddings[index + c], grad_embeddings_cur[cache_index * N_C + c]);
      }
    }
  }
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_second_backward_inputs(const float *__restrict__ inputs, const scalar_t *__restrict__ grid,
    const int *__restrict__ offsets, const scalar_t *__restrict__ grad, const float *__restrict__ grad_grad_inputs,
    float *__restrict__ grad2_inputs, const uint32_t B, const uint32_t L, const float S, const uint32_t H,
    const int valid_level, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
  const uint32_t b     = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t level = blockIdx.y;
  if (b >= B || level > valid_level) return;
  inputs += b * D;  // shape: [B, D]
// check input range in [0, 1]
#pragma unroll
  for (uint32_t d = 0; d < D; ++d) {
    if (inputs[d] < 0 || inputs[d] > 1.0f) return;  // grad is 0
  }
  grad2_inputs += b * D;          // shape: [B, D]
  grad_grad_inputs += b * D;      // shape: [B, D]
  grad += b * C + level * B * C;  // shape: [L, B, C]
  grid += offsets[level] * C;     // shape: [L, hashmap_size[l], C]

  const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
  const float scale           = exp2f(level * S) * H - 1.0f;
  const uint32_t resolution   = (uint32_t) ceil(scale) + 1;

  float pos[D];
  float pos_derivative[D];
  float pos_2nd_derivative[D] = {0};
  uint32_t pos_grid[D];

#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    pos[d]      = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
    pos_grid[d] = floorf(pos[d]);
    pos[d] -= (float) pos_grid[d];
    // smoothstep instead of linear
    if (interp == 1) {
      pos_2nd_derivative[d] = smoothstep_derivative_derivative(pos[d]);
      pos_derivative[d]     = smoothstep_derivative(pos[d]);
      pos[d]                = smoothstep(pos[d]);
    } else {
      pos_derivative[d] = 1.f;
    }
  }
  scalar_t grad_cur[C];
#pragma unroll
  for (uint32_t c = 0; c < C; ++c) grad_cur[c] = grad[c];

  float grad_in_diag[D];
  float grad_in_other[D];
#pragma unroll
  for (uint32_t d = 0; d < D; ++d) {
    grad_in_diag[d]  = scale * scale * grad_grad_inputs[d] * pos_2nd_derivative[d];  // from diagonal part of Hessian
    grad_in_other[d] = scale * scale * grad_grad_inputs[d] * pos_derivative[d];      // from other part of Hessian
  }

  auto calc_dLdx = [&](const uint32_t local_pos[D], const float weight) {
    uint32_t index  = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, local_pos);
    float dL_dx_dim = 0;
#pragma unroll
    for (uint32_t c = 0; c < C; ++c) {
      dL_dx_dim += (float) grid[index + c] * (float) grad_cur[c];
    }
    return dL_dx_dim * weight;
  };

#pragma unroll
  for (uint32_t gd = 0; gd < D; ++gd) {
    float grad_out = 0;
#pragma unroll
    for (uint32_t idx = 0; idx < (1 << (D - 1)); ++idx) {
      // Note There may be some bug, it can not pass gradient check
      if (interp == 1) {  // Note Linear Interpolations's diagnoal part is 0
        float weight_2nd_diag = grad_in_diag[gd];
        uint32_t pos_grid_local[D];
#pragma unroll
        for (uint32_t ngd = 0; ngd < D - 1; ++ngd) {
          const uint32_t dim = ngd >= gd ? (ngd + 1) : ngd;
          if ((idx & (1 << ngd)) == 0) {
            weight_2nd_diag *= 1 - pos[dim];
            pos_grid_local[dim] = pos_grid[dim];
          } else {
            weight_2nd_diag *= pos[dim];
            pos_grid_local[dim] = pos_grid[dim] + 1;
          }
        }
        // left
        pos_grid_local[gd] = pos_grid[gd];
        grad_out += calc_dLdx(pos_grid_local, -weight_2nd_diag);
        // right
        pos_grid_local[gd] = pos_grid[gd] + 1;
        grad_out += calc_dLdx(pos_grid_local, weight_2nd_diag);
      }
      if (D > 1) {
#pragma unroll
        for (uint32_t other_dim = 0; other_dim < D - 1; ++other_dim) {
          const uint32_t real_other_dim = other_dim >= gd ? (other_dim + 1) : other_dim;
          float weight_2nd_other        = grad_in_other[gd] * pos_derivative[gd];
          uint32_t pos_grid_local[D];
#pragma unroll
          for (uint32_t ngd = 0; ngd < D - 1; ++ngd) {
            const uint32_t dim = ngd >= real_other_dim ? (ngd + 1) : ngd;
            if ((idx & (1 << ngd)) == 0) {
              weight_2nd_other *= dim == gd ? -1.f : (1.f - pos[dim]);
              pos_grid_local[dim] = pos_grid[dim];
            } else {
              if (dim != gd) weight_2nd_other *= pos[dim];
              pos_grid_local[dim] = pos_grid[dim] + 1;
            }
          }

          // left
          pos_grid_local[real_other_dim] = pos_grid[real_other_dim];
          grad_out += calc_dLdx(pos_grid_local, -weight_2nd_other);
          // right
          pos_grid_local[real_other_dim] = pos_grid[real_other_dim] + 1;
          grad_out += calc_dLdx(pos_grid_local, weight_2nd_other);
        }
      }
    }
    atomicAdd(grad2_inputs + gd, grad_out);
  }
}

#define GRID_ENCODER_EXPAND_C(func, D, ...)                                \
  switch (C) {                                                             \
    case 1: func<scalar_t, D, 1> __VA_ARGS__; break;                       \
    case 2: func<scalar_t, D, 2> __VA_ARGS__; break;                       \
    case 4: func<scalar_t, D, 4> __VA_ARGS__; break;                       \
    case 8: func<scalar_t, D, 8> __VA_ARGS__; break;                       \
    default: throw std::runtime_error{#func ": C must be 1, 2, 4, or 8."}; \
  }

#define GRID_ENCODER_EXPAND_D(func, ...)                                \
  switch (D) {                                                          \
    case 2: GRID_ENCODER_EXPAND_C(func, 2, __VA_ARGS__); break;         \
    case 3: GRID_ENCODER_EXPAND_C(func, 3, __VA_ARGS__); break;         \
    case 4: GRID_ENCODER_EXPAND_C(func, 4, __VA_ARGS__); break;         \
    case 5: GRID_ENCODER_EXPAND_C(func, 5, __VA_ARGS__); break;         \
    default: throw std::runtime_error{#func ": D must be 2, 3, 4, 5."}; \
  }

void grid_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets,
    at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S,
    const uint32_t H, at::optional<at::Tensor> dy_dx, const uint32_t valid_level, const uint32_t gridtype,
    const bool align_corners, const uint32_t interp) {
  CHECK_CUDA(inputs);
  CHECK_CUDA(embeddings);
  CHECK_CUDA(offsets);
  CHECK_CUDA(outputs);
  // CHECK_CUDA(dy_dx);

  CHECK_CONTIGUOUS(inputs);
  CHECK_CONTIGUOUS(embeddings);
  CHECK_CONTIGUOUS(offsets);
  CHECK_CONTIGUOUS(outputs);
  // CHECK_CONTIGUOUS(dy_dx);

  CHECK_IS_FLOATING(inputs);
  CHECK_IS_FLOATING(embeddings);
  CHECK_IS_INT(offsets);
  CHECK_IS_FLOATING(outputs);
  // CHECK_IS_FLOATING(dy_dx);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(embeddings.scalar_type(), "grid_encode_forward", ([&] {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid         = {div_round_up(B, N_THREAD), L, 1};
    GRID_ENCODER_EXPAND_D(kernel_grid_encode_forward,
        KERNEL_ARG(blocks_hashgrid, N_THREAD)(inputs.data_ptr<float>(), embeddings.data_ptr<scalar_t>(),
            offsets.data_ptr<int>(), outputs.data_ptr<scalar_t>(), B, L, S, H,
            dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr, valid_level, gridtype, align_corners,
            interp));
  }));
}

void grid_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings,
    const at::Tensor offsets, const at::optional<at::Tensor> dy_dx, at::optional<at::Tensor> grad_inputs,
    at::optional<at::Tensor> grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L,
    const float S, const uint32_t H, const uint32_t valid_level, const uint32_t gridtype, const bool align_corners,
    const uint32_t interp) {
  CHECK_CUDA(grad);
  CHECK_CUDA(inputs);
  CHECK_CUDA(embeddings);
  CHECK_CUDA(offsets);

  CHECK_CONTIGUOUS(grad);
  CHECK_CONTIGUOUS(inputs);
  CHECK_CONTIGUOUS(embeddings);
  CHECK_CONTIGUOUS(offsets);

  CHECK_IS_FLOATING(grad);
  CHECK_IS_FLOATING(inputs);
  CHECK_IS_FLOATING(embeddings);
  CHECK_IS_INT(offsets);

  if (grad_embeddings.has_value()) {
    CHECK_CUDA(grad_embeddings.value());
    CHECK_CONTIGUOUS(grad_embeddings.value());
    CHECK_IS_FLOATING(grad_embeddings.value());
  }

  if (dy_dx.has_value()) {
    CHECK_CUDA(dy_dx.value());
    CHECK_CONTIGUOUS(dy_dx.value());
    CHECK_IS_FLOATING(dy_dx.value());
  }

  if (grad_inputs.has_value()) {
    CHECK_CUDA(grad_inputs.value());
    CHECK_CONTIGUOUS(grad_inputs.value());
    CHECK_IS_FLOATING(grad_inputs.value());
  }
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "grid_encode_backward", ([&] {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t N_C                 = std::min(2u, C);  // n_features_per_thread
    const dim3 blocks_hashgrid         = {div_round_up(B * C / N_C, N_THREAD), L, 1};

    if (grad_embeddings.has_value()) {
      GRID_ENCODER_EXPAND_D(kernel_grid_backward,
          KERNEL_ARG(blocks_hashgrid, N_THREAD)(grad.data_ptr<scalar_t>(), inputs.data_ptr<float>(),
              embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_embeddings.value().data_ptr<scalar_t>(), B,
              L, S, H, valid_level, gridtype, align_corners, interp));
      CHECK_CUDA_ERROR("kernel_grid_backward");
    }

    if (grad_inputs.has_value()) {
      BCNN_ASSERT(dy_dx.has_value(), "grad_inputs must not be None");
      GRID_ENCODER_EXPAND_D(kernel_input_backward,
          KERNEL_ARG(div_round_up(B * D, N_THREAD), N_THREAD)(grad.data_ptr<scalar_t>(),
              dy_dx.value().data_ptr<scalar_t>(), grad_inputs.value().data_ptr<scalar_t>(), B, L));
      CHECK_CUDA_ERROR("kernel_input_backward");
    }
  }));
}

void grid_encode_second_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings,
    const at::Tensor offsets, const at::Tensor dy_dx, const at::Tensor grad_grad_inputs, at::optional<Tensor> grad_grad,
    at::optional<Tensor> grad2_inputs, at::optional<Tensor> grad2_embeddings, const uint32_t B, const uint32_t D,
    const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t valid_level,
    const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
  CHECK_CUDA(grad);
  CHECK_CUDA(inputs);
  CHECK_CUDA(embeddings);
  CHECK_CUDA(offsets);
  CHECK_CUDA(dy_dx);
  CHECK_CUDA(grad_grad_inputs);

  CHECK_CONTIGUOUS(grad);
  CHECK_CONTIGUOUS(inputs);
  CHECK_CONTIGUOUS(embeddings);
  CHECK_CONTIGUOUS(offsets);
  CHECK_CONTIGUOUS(dy_dx);
  CHECK_CONTIGUOUS(grad_grad_inputs);

  CHECK_IS_FLOATING(grad);
  CHECK_IS_FLOATING(inputs);
  CHECK_IS_FLOATING(embeddings);
  CHECK_IS_INT(offsets);
  CHECK_IS_FLOATING(dy_dx);
  CHECK_IS_FLOATING(grad_grad_inputs);

  if (grad_grad.has_value()) {
    CHECK_CONTIGUOUS(grad_grad.value());
    CHECK_CUDA(grad_grad.value());
    CHECK_IS_FLOATING(grad_grad.value());
  }
  if (grad2_inputs.has_value()) {
    CHECK_CONTIGUOUS(grad2_inputs.value());
    CHECK_CUDA(grad2_inputs.value());
    CHECK_IS_FLOATING(grad2_inputs.value());
  }

  if (grad2_embeddings.has_value()) {
    CHECK_CONTIGUOUS(grad2_embeddings.value());
    CHECK_CUDA(grad2_embeddings.value());
    CHECK_IS_FLOATING(grad2_embeddings.value());
  }
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(embeddings.scalar_type(), "grid_encode_second_backward", ([&] {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t N_C                 = std::min(2u, C);  // n_features_per_thread
    const dim3 blocks_hashgrid         = {div_round_up(B * C / N_C, N_THREAD), L, 1};
    if (grad_grad.has_value()) {
      GRID_ENCODER_EXPAND_D(kernel_second_backward_grad,
          KERNEL_ARG(blocks_hashgrid, N_THREAD)(grad.data_ptr<scalar_t>(), inputs.data_ptr<float>(),
              embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_grad_inputs.data_ptr<float>(),
              dy_dx.data_ptr<scalar_t>(), grad_grad.value().data_ptr<scalar_t>(), B, L, S, H, valid_level, gridtype,
              align_corners, interp));
      cudaDeviceSynchronize();
      CHECK_CUDA_ERROR("kernel_second_backward_grad");
    }
    if (grad2_embeddings.has_value()) {
      GRID_ENCODER_EXPAND_D(kernel_second_backward_embedding,
          KERNEL_ARG(blocks_hashgrid, N_THREAD)(grad.data_ptr<scalar_t>(), inputs.data_ptr<float>(),
              embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_grad_inputs.data_ptr<float>(),
              grad2_embeddings.value().data_ptr<scalar_t>(), B, L, S, H, valid_level, gridtype, align_corners, interp));
      cudaDeviceSynchronize();
      CHECK_CUDA_ERROR("kernel_second_backward_embedding");
    }
    static constexpr uint32_t N_THREAD_2 = 512;
    const dim3 blocks_hashgrid_2         = {div_round_up(B, N_THREAD_2), L, 1};
    if (grad2_inputs.has_value()) {
      GRID_ENCODER_EXPAND_D(kernel_second_backward_inputs,
          KERNEL_ARG(blocks_hashgrid_2, N_THREAD_2)(inputs.data_ptr<float>(), embeddings.data_ptr<scalar_t>(),
              offsets.data_ptr<int>(), grad.data_ptr<scalar_t>(), grad_grad_inputs.data_ptr<float>(),
              grad2_inputs.value().data<float>(), B, L, S, H, valid_level, gridtype, align_corners, interp));
      CHECK_CUDA_ERROR("kernel_second_backward_inputs");
    }
  }));
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grad_tv(const float *__restrict__ inputs, const scalar_t *__restrict__ grid,
    scalar_t *__restrict__ grad, const int *__restrict__ offsets, const float weight, const uint32_t B,
    const uint32_t L, const float S, const uint32_t H, const uint32_t valid_level, const uint32_t gridtype,
    const bool align_corners) {
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= B) return;

  const uint32_t level = blockIdx.y;
  if (level > valid_level) return;
  // locate
  inputs += b * D;
  grid += (uint32_t) offsets[level] * C;
  grad += (uint32_t) offsets[level] * C;

  // check input range (should be in [0, 1])
  bool flag_oob = false;
#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    if (inputs[d] < 0 || inputs[d] > 1) {
      flag_oob = true;
    }
  }

  // if input out of bound, do nothing
  if (flag_oob) return;

  const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
  const float scale           = exp2f(level * S) * H - 1.0f;
  const uint32_t resolution   = (uint32_t) ceil(scale) + 1;

  // calculate coordinate
  float pos[D];
  uint32_t pos_grid[D];  // [0, resolution]

#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    pos[d]      = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
    pos_grid[d] = floorf(pos[d]);
    // pos[d] -= (float)pos_grid[d]; // not used
  }

  // total variation on pos_grid
  scalar_t results[C] = {0};  // temp results in register
  scalar_t idelta[C]  = {0};

  uint32_t index = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

  scalar_t w = weight / (2 * D);

#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    uint32_t cur_d = pos_grid[d];
    scalar_t grad_val;

    // right side
    if (cur_d < resolution) {
      pos_grid[d]          = cur_d + 1;
      uint32_t index_right = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

#pragma unroll
      for (uint32_t ch = 0; ch < C; ch++) {
        // results[ch] += w * clamp(grid[index + ch] - grid[index_right + ch], -1.0f, 1.0f);
        grad_val = (grid[index + ch] - grid[index_right + ch]);
        results[ch] += grad_val;
        idelta[ch] += grad_val * grad_val;
      }
    }

    // left side
    if (cur_d > 0) {
      pos_grid[d]         = cur_d - 1;
      uint32_t index_left = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

#pragma unroll
      for (uint32_t ch = 0; ch < C; ch++) {
        // results[ch] += w * clamp(grid[index + ch] - grid[index_left + ch], -1.0f, 1.0f);
        grad_val = (grid[index + ch] - grid[index_left + ch]);
        results[ch] += grad_val;
        idelta[ch] += grad_val * grad_val;
      }
    }

    // reset
    pos_grid[d] = cur_d;
  }

// writing to global memory (slow)
#pragma unroll
  for (uint32_t ch = 0; ch < C; ch++) {
    // index may collide, so use atomic!
    atomicAdd(&grad[index + ch], w * results[ch] * rsqrtf(idelta[ch] + 1e-9f));
  }
}

void grad_total_variation(const at::Tensor inputs, const at::Tensor embeddings, at::Tensor grad,
    const at::Tensor offsets, const float weight, const uint32_t B, const uint32_t D, const uint32_t C,
    const uint32_t L, const float S, const uint32_t H, const uint32_t valid_level, const uint32_t gridtype,
    const bool align_corners) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(embeddings.scalar_type(), "grad_total_variation", ([&] {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks_hashgrid         = {div_round_up(B, N_THREAD), L, 1};
    GRID_ENCODER_EXPAND_D(
        kernel_grad_tv, KERNEL_ARG(blocks_hashgrid, N_THREAD)(inputs.data_ptr<float>(), embeddings.data_ptr<scalar_t>(),
                            grad.data_ptr<scalar_t>(), offsets.data_ptr<int>(), weight, B, L, S, H, valid_level,
                            gridtype, align_corners))
  }));
}

REGIST_PYTORCH_EXTENSION(nerf_grid_encode, {
  m.def("grid_encode_forward", &grid_encode_forward, "grid_encode_forward (CUDA)");
  m.def("grid_encode_backward", &grid_encode_backward, "grid_encode_backward (CUDA)");
  m.def("grid_encode_second_backward", &grid_encode_second_backward, "grid_encode_second_backward (CUDA)");
  m.def("grad_total_variation", &grad_total_variation, "grad_total_variation (CUDA)");
})
