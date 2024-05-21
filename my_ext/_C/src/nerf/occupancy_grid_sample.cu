#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/torch.h>

#include <cstdio>
#include <limits>
#include <stdexcept>

#include "util.cuh"

inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }

inline __host__ __device__ float signf(const float x) { return copysignf(1.0, x); }

inline __host__ __device__ void swapf(float& a, float& b) {
  float c = a;
  a       = b;
  b       = c;
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
  const float mx = fmaxf(fabsf(x), fmaxf(fabsf(y), fabsf(z)));
  int exponent;
  frexpf(mx, &exponent);  // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, [2, 4) --> 2, ...
  return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __device__ int mip_from_dt(const float dt, const float H, const float max_cascade) {
  const float mx = dt * H * 0.5;
  int exponent;
  frexpf(mx, &exponent);
  return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __host__ __device__ uint32_t __expand_bits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t xx = __expand_bits(x);
  uint32_t yy = __expand_bits(y);
  uint32_t zz = __expand_bits(z);
  return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x) {
  x = x & 0x49249249;
  x = (x | (x >> 2)) & 0xc30c30c3;
  x = (x | (x >> 4)) & 0x0f00f00f;
  x = (x | (x >> 8)) & 0xff0000ff;
  x = (x | (x >> 16)) & 0x0000ffff;
  return x;
}

////////////////////////////////////////////////////
/////////////           utils          /////////////
////////////////////////////////////////////////////

// coords: int32, [N, 3]
// indices: int32, [N]
__global__ void kernel_morton3D(const int* __restrict__ coords, const uint32_t N, int* indices) {
  // parallel
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  // locate
  coords += n * 3;
  indices[n] = __morton3D(coords[0], coords[1], coords[2]);
}

void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices) {
  static constexpr uint32_t N_THREAD = 128;
  kernel_morton3D KERNEL_ARG(div_round_up(N, N_THREAD), N_THREAD)(coords.data_ptr<int>(), N, indices.data_ptr<int>());
}

// indices: int32, [N]
// coords: int32, [N, 3]
__global__ void kernel_morton3D_invert(const int* __restrict__ indices, const uint32_t N, int* coords) {
  // parallel
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  // locate
  coords += n * 3;

  const int ind = indices[n];

  coords[0] = __morton3D_invert(ind >> 0);
  coords[1] = __morton3D_invert(ind >> 1);
  coords[2] = __morton3D_invert(ind >> 2);
}

void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords) {
  static constexpr uint32_t N_THREAD = 128;
  kernel_morton3D_invert KERNEL_ARG(div_round_up(N, N_THREAD), N_THREAD)(
      indices.data_ptr<int>(), N, coords.data_ptr<int>());
}

// grid: float, [C, H, H, H]
// N: int, C * H * H * H / 8
// density_thresh: float
// bitfield: uint8, [N]
template <typename scalar_t>
__global__ void kernel_packbits(
    const scalar_t* __restrict__ grid, const uint32_t N, const float density_thresh, uint8_t* bitfield) {
  // parallel per byte
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  // locate
  grid += n * 8;

  uint8_t bits = 0;

#pragma unroll
  for (uint8_t i = 0; i < 8; i++) {
    bits |= (grid[i] > density_thresh) ? ((uint8_t) 1 << i) : 0;
  }

  bitfield[n] = bits;
}

void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield) {
  static constexpr uint32_t N_THREAD = 128;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grid.scalar_type(), "packbits", ([&] {
    kernel_packbits KERNEL_ARG(div_round_up(N, N_THREAD), N_THREAD)(
        grid.data_ptr<scalar_t>(), N, density_thresh, bitfield.data_ptr<uint8_t>());
  }));
}

__global__ void kernel_flatten_rays(const int* __restrict__ rays, const uint32_t N, const uint32_t M, int* res) {
  // parallel per ray
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  // locate
  uint32_t offset    = rays[n * 2];
  uint32_t num_steps = rays[n * 2 + 1];

  // write to res
  res += offset;
  for (int i = 0; i < num_steps; i++) res[i] = n;
}

void flatten_rays(const at::Tensor rays, const uint32_t N, const uint32_t M, at::Tensor res) {
  static constexpr uint32_t N_THREAD = 128;

  kernel_flatten_rays KERNEL_ARG(div_round_up(N, N_THREAD), N_THREAD)(rays.data_ptr<int>(), N, M, res.data_ptr<int>());
}

////////////////////////////////////////////////////
/////////////         training         /////////////
////////////////////////////////////////////////////

// rays_o/d: [N, 3]
// grid: [CHHH / 8]
// xyzs, dirs, ts: [M, 3], [M, 3], [M, 2]
// dirs: [M, 3]
// rays: [N, 3], idx, offset, num_steps
template <typename scalar_t>
__global__ void kernel_march_rays_train(
    // options
    const bool contract, const float dt_gamma, const uint32_t max_steps, const uint32_t N, const uint32_t C,
    const uint32_t H,
    // inputs
    const scalar_t* __restrict__ rays_o, const scalar_t* __restrict__ rays_d, const uint8_t* __restrict__ grid,
    const scalar_t* __restrict__ t_range, const scalar_t* __restrict__ noises, const float bound,
    // outputs
    int* __restrict__ ray_indices, scalar_t* __restrict__ time, int* __restrict__ pack_info) {
  // parallel per ray
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;
  const float near = t_range[n * 2];
  const float far  = t_range[n * 2 + 1];
  if (near < 0 or near > far) return;  // invalid t

  // is first pass running.
  const bool first_pass = (ray_indices == nullptr);

  // locate
  rays_o += n * 3;
  rays_d += n * 3;

  uint32_t num_steps = max_steps;

  const float dt_min = 2 * SQRT3() / max_steps;
  const float dt_max = 2 * SQRT3() * bound / H;
  // const float dt_max = 1e10f;
  if (!first_pass) {
    size_t offset = pack_info[n * 2];
    num_steps     = pack_info[n * 2 + 1];
    ray_indices += offset * 2;
    time += offset * 2;
  }

  // ray marching
  const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
  const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
  const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
  const float rH = 1 / (float) H;
  const float H3 = H * H * H;

  const float noise = noises == nullptr ? 1.f : (float) noises[n];

  float t0 = near;
  t0 += clamp(t0 * dt_gamma, dt_min, dt_max) * noise;
  float t       = t0;
  uint32_t step = 0;

  // if (t < far) printf("valid ray %d t=%f near=%f far=%f \n", n, t, near, far);

  while (t < far && step < num_steps) {
    // current point
    const float x = clamp(ox + t * dx, -bound, bound);
    const float y = clamp(oy + t * dy, -bound, bound);
    const float z = clamp(oz + t * dz, -bound, bound);

    float dt = clamp(t * dt_gamma, dt_min, dt_max);

    // get mip level
    const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C));  // range in [0, C - 1]

    const float mip_bound  = fminf(scalbnf(1.0f, level), bound);
    const float mip_rbound = 1 / mip_bound;

    // contraction
    float cx = x, cy = y, cz = z;
    const float mag = fmaxf(fabsf(x), fmaxf(fabsf(y), fabsf(z)));
    if (contract && mag > 1) {
      // L-INF norm
      const float Linf_scale = (2 - 1 / mag) / mag;
      cx *= Linf_scale;
      cy *= Linf_scale;
      cz *= Linf_scale;
    }

    // convert to nearest grid position
    const int nx = clamp(0.5f * (cx * mip_rbound + 1) * H, 0.0f, (float) (H - 1));
    const int ny = clamp(0.5f * (cy * mip_rbound + 1) * H, 0.0f, (float) (H - 1));
    const int nz = clamp(0.5f * (cz * mip_rbound + 1) * H, 0.0f, (float) (H - 1));

    const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
    const bool occ       = grid[index / 8] & (1 << (index % 8));

    // if occpuied, advance a small step, and write to output
    // if (n == 0) printf("t=%f density=%f vs thresh=%f step=%d\n", t, density, density_thresh, step);

    if (occ || (contract && mag > 1)) {
      t += dt;
      if (!first_pass) {
        *(ray_indices++) = n;
        *(ray_indices++) = step;
        time[0]          = t;
        time[1]          = dt;
        time += 2;
      }
      step++;
      // contraction case: cannot apply voxel skipping.
      // } else if (contract && mag > 1) {
      //   t += dt;
      // else, skip a large step (basically skip a voxel grid)
    } else {
      // calc distance to next voxel
      const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - cx) * rdx;
      const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - cy) * rdy;
      const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - cz) * rdz;

      const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
      // step until next voxel
      do {
        dt = clamp(t * dt_gamma, dt_min, dt_max);
        t += dt;
      } while (t < tt);
    }
  }

  // write rays
  if (first_pass) {
    int point_index      = atomicAdd(pack_info + N * 2, (int) step);
    pack_info[n * 2]     = point_index;
    pack_info[n * 2 + 1] = step;
  }
}

vector<Tensor> march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor t_range,
    const at::Tensor occ_grid, const float bound, const bool contract, const float dt_gamma, const uint32_t max_steps,
    const uint32_t C, const uint32_t H, at::optional<Tensor> noises, at::optional<Tensor> pack_info) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(t_range);
  CHECK_INPUT_AND_TYPE(occ_grid, torch::kUInt8);
  BCNN_ASSERT(rays_o.ndimension() == 2 && rays_o.sizes() == rays_d.sizes(), "Error shape for rays_o/rays_d");
  const uint32_t N = rays_o.size(0);
  BCNN_ASSERT(t_range.ndimension() == 2 && t_range.size(0) == N && t_range.size(1) == 2, "Error shape for t_range");

  static constexpr uint32_t N_THREAD = 128;
  Tensor info, ray_indices, ts;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.scalar_type(), "march_rays_train", ([&] {
    if (!pack_info.has_value()) {  // first pass
      info = torch::zeros({N + 1, 2}, rays_o.options().dtype(torch::kInt32));
      kernel_march_rays_train<scalar_t> KERNEL_ARG(div_round_up(N, N_THREAD), N_THREAD)(contract, dt_gamma, max_steps,
          N, C, H, rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), occ_grid.data_ptr<uint8_t>(),
          t_range.data_ptr<scalar_t>(), noises.has_value() ? noises.value().data_ptr<scalar_t>() : nullptr, bound,
          nullptr, nullptr, info.data_ptr<int>());
      CHECK_CUDA_ERROR("kernel_march_rays_train first pass");
    } else {
      info = pack_info.value();
      CHECK_INPUT_AND_TYPE(info, torch::kInt32);
      BCNN_ASSERT(info.ndimension() == 2 && info.size(0) == N && info.size(1) == 2, "Error shape for pack_info");
    }
    int M       = info[N][0].item<int>();
    ray_indices = torch::empty({M, 2}, rays_o.options().dtype(torch::kInt32));
    ts          = torch::empty({M, 2}, rays_o.options());
    kernel_march_rays_train<scalar_t> KERNEL_ARG(div_round_up(N, N_THREAD), N_THREAD)(contract, dt_gamma, max_steps, N,
        C, H, rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), occ_grid.data_ptr<uint8_t>(),
        t_range.data_ptr<scalar_t>(), noises.has_value() ? noises.value().data_ptr<scalar_t>() : nullptr, bound,
        ray_indices.data_ptr<int>(), ts.data_ptr<scalar_t>(), info.data_ptr<int>());
    CHECK_CUDA_ERROR("kernel_march_rays_train");
  }));
  return {ray_indices, ts, info};
}

////////////////////////////////////////////////////
/////////////          infernce        /////////////
////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void kernel_march_rays(const uint32_t n_alive, const uint32_t n_step, const int* __restrict__ rays_alive,
    const scalar_t* __restrict__ rays_t, const scalar_t* __restrict__ rays_o, const scalar_t* __restrict__ rays_d,
    const float bound, const bool contract, const float dt_gamma, const uint32_t max_steps, const uint32_t C,
    const uint32_t H, const uint8_t* __restrict__ grid, const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars, scalar_t* xyzs, scalar_t* dirs, scalar_t* ts,
    const scalar_t* __restrict__ noises) {
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= n_alive) return;

  const int index   = rays_alive[n];  // ray id
  const float noise = noises[n];

  // locate
  rays_o += index * 3;
  rays_d += index * 3;
  xyzs += n * n_step * 3;
  dirs += n * n_step * 3;
  ts += n * n_step * 2;

  const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
  const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
  const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
  const float rH = 1 / (float) H;
  const float H3 = H * H * H;

  const float near = nears[index], far = fars[index];

  const float dt_min = 2 * SQRT3() / max_steps;
  const float dt_max = 2 * SQRT3() * bound / H;
  // const float dt_max = 1e10f;

  // march for n_step steps, record points
  float t = rays_t[index];
  t += clamp(t * dt_gamma, dt_min, dt_max) * noise;
  uint32_t step = 0;

  while (t < far && step < n_step) {
    // current point
    const float x = clamp(ox + t * dx, -bound, bound);
    const float y = clamp(oy + t * dy, -bound, bound);
    const float z = clamp(oz + t * dz, -bound, bound);

    float dt = clamp(t * dt_gamma, dt_min, dt_max);

    // get mip level
    const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C));  // range in [0, C - 1]

    const float mip_bound  = fminf(scalbnf(1, level), bound);
    const float mip_rbound = 1 / mip_bound;

    // contraction
    float cx = x, cy = y, cz = z;
    const float mag = fmaxf(fabsf(x), fmaxf(fabsf(y), fabsf(z)));
    if (contract && mag > 1) {
      // L-INF norm
      const float Linf_scale = (2 - 1 / mag) / mag;
      cx *= Linf_scale;
      cy *= Linf_scale;
      cz *= Linf_scale;
    }

    // convert to nearest grid position
    const int nx = clamp(0.5f * (cx * mip_rbound + 1) * H, 0.0f, (float) (H - 1));
    const int ny = clamp(0.5f * (cy * mip_rbound + 1) * H, 0.0f, (float) (H - 1));
    const int nz = clamp(0.5f * (cz * mip_rbound + 1) * H, 0.0f, (float) (H - 1));

    const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
    const bool occ       = grid[index / 8] & (1 << (index % 8));

    // if occpuied, advance a small step, and write to output
    if (occ) {
      // write step
      xyzs[0] = cx;
      xyzs[1] = cy;
      xyzs[2] = cz;
      dirs[0] = dx;
      dirs[1] = dy;
      dirs[2] = dz;
      // calc dt
      t += dt;
      ts[0] = t;
      ts[1] = dt;
      // step
      xyzs += 3;
      dirs += 3;
      ts += 2;
      step++;

      // contraction case
    } else if (contract && mag > 1) {
      t += dt;
      // else, skip a large step (basically skip a voxel grid)
    } else {
      // calc distance to next voxel
      const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - cx) * rdx;
      const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - cy) * rdy;
      const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - cz) * rdz;
      const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
      // step until next voxel
      do {
        dt = clamp(t * dt_gamma, dt_min, dt_max);
        t += dt;
      } while (t < tt);
    }
  }
}

void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t,
    const at::Tensor rays_o, const at::Tensor rays_d, const float bound, const bool contract, const float dt_gamma,
    const uint32_t max_steps, const uint32_t C, const uint32_t H, const at::Tensor grid, const at::Tensor near,
    const at::Tensor far, at::Tensor xyzs, at::Tensor dirs, at::Tensor ts, at::Tensor noises) {
  static constexpr uint32_t N_THREAD = 128;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.scalar_type(), "march_rays", ([&] {
    kernel_march_rays KERNEL_ARG(div_round_up(n_alive, N_THREAD), N_THREAD)(n_alive, n_step, rays_alive.data_ptr<int>(),
        rays_t.data_ptr<scalar_t>(), rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), bound, contract,
        dt_gamma, max_steps, C, H, grid.data_ptr<uint8_t>(), near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(),
        xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), ts.data_ptr<scalar_t>(), noises.data_ptr<scalar_t>());
  }));
}

REGIST_PYTORCH_EXTENSION(occupancy_grid_sample, {
  // utils
  m.def("flatten_rays", &flatten_rays, "flatten_rays (CUDA)");
  m.def("packbits", &packbits, "packbits (CUDA)");
  m.def("morton3D", &morton3D, "morton3D (CUDA)");
  m.def("morton3D_invert", &morton3D_invert, "morton3D_invert (CUDA)");
  // train
  m.def("march_rays_train", &march_rays_train, "march_rays_train (CUDA)");
  // infer
  m.def("march_rays", &march_rays, "march rays (CUDA)");
});