#include "util.cuh"

inline constexpr __device__ float RPI() { return 0.3183098861837907f; }

template <typename scalar_t>
inline __host__ __device__ void swapf(scalar_t& a, scalar_t& b) {
  scalar_t c = a;
  a          = b;
  b          = c;
}

// rays_o/d: [N, 3]
// nears/fars: [N]
// scalar_t should always be float in use.
template <typename scalar_t>
__global__ void kernel_near_far_from_aabb(const scalar_t* __restrict__ rays_o, const scalar_t* __restrict__ rays_d,
    const scalar_t* __restrict__ aabb, const uint32_t N, const float min_near, const float eps,
    const scalar_t invalid_value, scalar_t* __restrict__ t_range) {
  // parallel per ray
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  // locate
  rays_o += n * 3;
  rays_d += n * 3;
  t_range += n * 2;

  const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
  const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
  const float rdx = 1.f / (dx + eps), rdy = 1.f / (dy + eps), rdz = 1.f / (dz + eps);

  // get near far (assume cube scene)
  float near = (aabb[0] - ox) * rdx;
  float far  = (aabb[3] - ox) * rdx;
  if (near > far) swapf(near, far);

  float near_y = (aabb[1] - oy) * rdy;
  float far_y  = (aabb[4] - oy) * rdy;
  if (near_y > far_y) swapf(near_y, far_y);

  if (near_y > near) near = near_y;
  if (far_y < far) far = far_y;

  float near_z = (aabb[2] - oz) * rdz;
  float far_z  = (aabb[5] - oz) * rdz;
  if (near_z > far_z) swapf(near_z, far_z);

  if (near_z > near) near = near_z;
  if (far_z < far) far = far_z;

  if (near < min_near) near = min_near;

  if (near > far) {
    t_range[0] = t_range[1] = invalid_value;

  } else {
    t_range[0] = near;
    t_range[1] = far;
  }
}

Tensor near_far_from_aabb(
    Tensor rays_o, Tensor rays_d, Tensor aabb, const float min_near, const float eps, const float invalid_value) {
  BCNN_ASSERT(rays_o.device() == rays_d.device() && rays_o.device() == aabb.device(),
      "rays_o, rays_d, aabb must have same device");
  BCNN_ASSERT(rays_o.scalar_type() == rays_d.scalar_type() && rays_o.scalar_type() == aabb.scalar_type(),
      "rays_o, rays_d, aabb must have same dtype");
  BCNN_ASSERT(aabb.numel() == 6, "Error shape for aabb");

  rays_o = rays_o.contiguous();
  rays_d = rays_d.contiguous();
  aabb   = aabb.contiguous();

  auto shape              = rays_o.sizes().vec();
  shape[shape.size() - 1] = 2;
  Tensor t_range          = torch::empty(shape, rays_o.options());
  const uint32_t N        = t_range.numel() / 2;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.scalar_type(), "near_far_from_aabb", ([&] {
    const scalar_t* o_ptr    = rays_o.data_ptr<scalar_t>();
    const scalar_t* d_ptr    = rays_d.data_ptr<scalar_t>();
    const scalar_t* aabb_ptr = aabb.data_ptr<scalar_t>();
    if (rays_o.is_cuda()) {
      static constexpr uint32_t N_THREAD = 128;
      kernel_near_far_from_aabb<scalar_t> KERNEL_ARG(div_round_up(N, N_THREAD), N_THREAD)(
          o_ptr, d_ptr, aabb_ptr, N, min_near, eps, invalid_value, t_range.data<scalar_t>());
      CHECK_CUDA_ERROR("kernel_near_far_from_aabb");
    } else {
      scalar_t* ptr = t_range.data_ptr<scalar_t>();
      for (uint32_t i = 0; i < N; ++i) {
        float tmp   = 1.f / (d_ptr[i * 3 + 0] + eps);
        float x_min = (aabb_ptr[0] - o_ptr[i * 3 + 0]) * tmp, x_max = (aabb_ptr[3] - o_ptr[i * 3 + 0]) * tmp;
        if (x_min > x_max) swapf(x_min, x_max);

        tmp         = 1.f / (d_ptr[i * 3 + 1] + eps);
        float y_min = (aabb_ptr[1] - o_ptr[i * 3 + 1]) * tmp, y_max = (aabb_ptr[4] - o_ptr[i * 3 + 1]) * tmp;
        if (y_min > y_max) swapf(y_min, y_max);
        if (x_min < y_min) x_min = y_min;
        if (x_max > y_max) x_max = y_max;

        tmp         = 1.f / (d_ptr[i * 3 + 2] + eps);
        float z_min = (aabb_ptr[2] - o_ptr[i * 3 + 2]) * tmp, z_max = (aabb_ptr[5] - o_ptr[i * 3 + 2]) * tmp;
        if (z_min > z_max) swapf(z_min, z_max);
        if (x_min < z_min) x_min = z_min;
        if (x_max > z_max) x_max = z_max;

        if (x_min < min_near) x_min = min_near;
        if (x_min > x_max) x_min = invalid_value, x_max = invalid_value;
        ptr[i * 2 + 0] = x_min;
        ptr[i * 2 + 1] = x_max;
      }
    }
  }));
  return t_range;
}

// rays_o/d: [N, 3]
// radius: float
// coords: [N, 2]
template <typename scalar_t>
__global__ void kernel_sph_from_ray(const scalar_t* __restrict__ rays_o, const scalar_t* __restrict__ rays_d,
    const float radius, const uint32_t N, scalar_t* coords) {
  // parallel per ray
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  // locate
  rays_o += n * 3;
  rays_d += n * 3;
  coords += n * 2;

  const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
  const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
  // const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

  // solve t from || o + td || = radius
  const float A = dx * dx + dy * dy + dz * dz;
  const float B = ox * dx + oy * dy + oz * dz;  // in fact B / 2
  const float C = ox * ox + oy * oy + oz * oz - radius * radius;

  const float t = (-B + sqrtf(B * B - A * C)) / A;  // always use the larger solution (positive)

  // solve theta, phi (assume y is the up axis)
  const float x = ox + t * dx, y = oy + t * dy, z = oz + t * dz;
  const float theta = atan2(sqrtf(x * x + z * z), y);  // [0, PI)
  const float phi   = atan2(z, x);                     // [-PI, PI)

  // normalize to [-1, 1]
  coords[0] = 2 * theta * RPI() - 1;
  coords[1] = phi * RPI();
}

void sph_from_ray(
    const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords) {
  static constexpr uint32_t N_THREAD = 128;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.scalar_type(), "sph_from_ray", ([&] {
    kernel_sph_from_ray KERNEL_ARG(div_round_up(N, N_THREAD), N_THREAD)(
        rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), radius, N, coords.data_ptr<scalar_t>());
  }));
}

REGIST_PYTORCH_EXTENSION(scene_bound, {
  m.def("near_far_from_aabb", &near_far_from_aabb, py::arg("rays_o"), py::arg("rays_d"), py::arg("aabb"),
      py::arg("min_near") = 0.2f, py::arg("eps") = 1e-15f, py::arg("invalid_value") = -1.f,
      "near_far_from_aabb (CUDA, CPU)");
  m.def("sph_from_ray", &sph_from_ray, "sph_from_ray (CUDA)");
});