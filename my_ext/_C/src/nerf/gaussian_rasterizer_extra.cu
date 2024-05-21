#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "util.cuh"

namespace cg = cooperative_groups;
namespace GaussianRasterizer {

template <int E_SPLIT = 16>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) render_extra_forward_kernel(int W, int H, int E,
    const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
    const float2* __restrict__ points_xy_image, const float4* __restrict__ conic_opacity,
    const uint32_t* __restrict__ n_contrib, const float* __restrict__ point_extra, float* __restrict__ pixel_extra) {
  // Identify current tile and associated min/max pixel range.
  auto block                       = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  const uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  const uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  const uint32_t pix_id            = W * pix.y + pix.x;
  const float2 pixf                = {(float) pix.x, (float) pix.y};
  pixel_extra += pix_id * E;

  const bool inside = pix.x < W && pix.y < H;
  const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  float temp_extra[E_SPLIT] = {0};
  // Iterate over batches until all done or range is complete
  for (int e_start = 0; e_start < E; e_start += E_SPLIT) {
    bool done            = !inside;
    int toDo             = range.y - range.x;
    float T              = 1.0f;
    uint32_t contributor = 0;
#pragma unroll
    for (int e = 0; e < E_SPLIT; ++e) temp_extra[e] = 0;

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
      int num_done = __syncthreads_count(done);
      if (num_done == BLOCK_SIZE) break;

      // Collectively fetch per-Gaussian data from global to shared
      int progress = i * BLOCK_SIZE + block.thread_rank();
      if (range.x + progress < range.y) {
        int coll_id                                  = point_list[range.x + progress];
        collected_id[block.thread_rank()]            = coll_id;
        collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
        collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
      }
      block.sync();

      // Iterate over current batch
      for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
        // Keep track of current position in range
        contributor++;
        if (contributor > last_contributor) {
          done = true;
          continue;
        }

        // Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
        float2 xy    = collected_xy[j];
        float2 d     = {xy.x - pixf.x, xy.y - pixf.y};
        float4 con_o = collected_conic_opacity[j];
        float power  = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f) continue;

        // Eq. (2) from 3D Gaussian splatting paper.
        // Obtain alpha by multiplying with Gaussian opacity and its exponential falloff from mean.
        // Avoid numerical instabilities (see paper appendix).
        float alpha = min(0.99f, con_o.w * exp(power));
        if (alpha < 1.0f / 255.0f) continue;
        float test_T = T * (1 - alpha);
        if (test_T < 0.0001f) {
          done = true;
          continue;
        }
#pragma unroll
        for (int e = 0; e < E_SPLIT; ++e)
          temp_extra[e] += (e_start + e < E) ? (point_extra[collected_id[j] * E + e + e_start] * alpha * T) : 0;
        T = test_T;
      }
    }
    if (inside) {
#pragma unroll
      for (int e = 0; e < E_SPLIT; ++e)
        if (e + e_start < E) pixel_extra[e + e_start] = temp_extra[e];
    }
  }
}

template <int E_SPLIT = 16>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) render_extra_backward_kernel(int W, int H, int E,
    const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
    const float2* __restrict__ points_xy_image, const float4* __restrict__ conic_opacity,
    const float* __restrict__ out_opacity, const uint32_t* __restrict__ n_contrib,
    const float* __restrict__ point_extra, const float* __restrict__ dL_dpixel_extra, float3* __restrict__ dL_dmean2D,
    float4* __restrict__ dL_dconic2D, float* __restrict__ dL_dopacity, float* __restrict__ dL_dpoint_extra) {
  // We rasterize again. Compute necessary block info.
  auto block                       = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  const uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  const uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  const uint32_t pix_id            = W * pix.y + pix.x;
  const float2 pixf                = {(float) pix.x, (float) pix.y};

  const bool inside = pix.x < W && pix.y < H;
  const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
  __shared__ float collected_extra[E_SPLIT * BLOCK_SIZE];

  const float T_final        = inside ? 1.0f - out_opacity[pix_id] : 0;
  const int last_contributor = inside ? n_contrib[pix_id] : 0;
  const float ddelx_dx       = 0.5 * W;
  const float ddely_dy       = 0.5 * H;

  for (int es = 0; es < E; es += E_SPLIT) {
    bool done                = !inside;
    int toDo                 = range.y - range.x;
    uint32_t contributor     = toDo;
    float T                  = T_final;
    float accum_rec[E_SPLIT] = {0};
    float dL_dpixel[E_SPLIT];
    if (inside) {
#pragma unroll
      for (int e = 0; e < E_SPLIT; e++) dL_dpixel[e] = (e + es < E) ? dL_dpixel_extra[pix_id * E + e + es] : 0;
    }
    float last_alpha          = 0;
    float last_extra[E_SPLIT] = {0};

    // Traverse all Gaussians
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
      // Load auxiliary data into shared memory, start in the BACK and load them in reverse order.
      block.sync();
      const int progress = i * BLOCK_SIZE + block.thread_rank();
      if (range.x + progress < range.y) {
        const int coll_id                            = point_list[range.y - progress - 1];
        collected_id[block.thread_rank()]            = coll_id;
        collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
        collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
#pragma unroll
        for (int e = 0; e < E_SPLIT; e++)
          collected_extra[e * BLOCK_SIZE + block.thread_rank()] = (e + es < E) ? point_extra[coll_id * E + e + es] : 0;
      }
      block.sync();

      // Iterate over Gaussians
      for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
        // Keep track of current Gaussian ID. Skip, if this one is behind the last contributor for this pixel.
        contributor--;
        if (contributor >= last_contributor) continue;

        // Compute blending values, as before.
        const float2 xy    = collected_xy[j];
        const float2 d     = {xy.x - pixf.x, xy.y - pixf.y};
        const float4 con_o = collected_conic_opacity[j];
        const float power  = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f) continue;

        const float G     = exp(power);
        const float alpha = min(0.99f, con_o.w * G);
        if (alpha < 1.0f / 255.0f) continue;

        T = T / (1.f - alpha);

        const float dchannel_dextra = alpha * T;

        // Propagate gradients to per-Gaussian colors and keep gradients w.r.t. alpha
        // (blending factor for a Gaussian/pixel pair).
        float dL_dalpha     = 0.0f;
        const int global_id = collected_id[j];
        for (int e = 0; e < E_SPLIT; e++) {
          if (e + es >= E) continue;
          const float c = collected_extra[e * BLOCK_SIZE + j];
          accum_rec[e]  = last_alpha * last_extra[e] + (1.f - last_alpha) * accum_rec[e];
          last_extra[e] = c;

          const float dL_dchannel = dL_dpixel[e];
          dL_dalpha += (c - accum_rec[e]) * dL_dchannel;
          atomicAdd(&(dL_dpoint_extra[global_id * E + e + es]), dchannel_dextra * dL_dchannel);
        }

        dL_dalpha *= T;
        last_alpha = alpha;

        // Helpful reusable temporary variables
        const float dL_dG    = con_o.w * dL_dalpha;
        const float gdx      = G * d.x;
        const float gdy      = G * d.y;
        const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
        const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

        // Update gradients w.r.t. 2D mean position of the Gaussian
        atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
        atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

        // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
        atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
        atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
        atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

        // Update gradients w.r.t. opacity of the Gaussian
        atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
      }
    }
  }
}

Tensor render_extra_forward(int width, int height, int num_rendered, const Tensor& extras, const Tensor& geomBuffer,
    const Tensor& binningBuffer, const Tensor& imageBuffer) {
  CHECK_INPUT(extras);
  BCNN_ASSERT(extras.ndimension() == 2, "Error shape for extras");
  int P              = extras.size(0);  // num points
  int E              = extras.size(1);  // num extras
  Tensor pixel_extra = torch::zeros({width, height, E}, extras.options());
  if (P == 0) return pixel_extra;

  char* geom_buffer         = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
  char* binning_buffer      = reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr());
  char* img_buffer          = reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr());
  GeometryState geomState   = GeometryState::fromChunk(geom_buffer, P);
  BinningState binningState = BinningState::fromChunk(binning_buffer, num_rendered);
  ImageState imgState       = ImageState::fromChunk(img_buffer, width * height);

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  render_extra_forward_kernel KERNEL_ARG(tile_grid, block)(width, height, E, imgState.ranges, binningState.point_list,
      geomState.means2D, geomState.conic_opacity, imgState.n_contrib, extras.contiguous().data_ptr<float>(),
      pixel_extra.data_ptr<float>());
  CHECK_CUDA_ERROR("render_extra_forward_kernel");
  return pixel_extra;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> render_extra_backward(int width, int height, int num_rendered,
    const Tensor& extras, const Tensor& out_opacity, const Tensor& grad_pixel_extras, const Tensor& geomBuffer,
    const Tensor& binningBuffer, const Tensor& imageBuffer, torch::optional<Tensor>& grad_means2D,
    torch::optional<Tensor>& grad_conic, torch::optional<Tensor>& grad_opacity) {
  int P = extras.size(0);  // num points
  int E = extras.size(1);  // num extras

  Tensor dL_dmeans2D = grad_means2D.has_value() ? grad_means2D.value() : torch::zeros({P, 3}, extras.options());
  Tensor dL_dconic   = grad_conic.has_value() ? grad_conic.value() : torch::zeros({P, 2, 2}, extras.options());
  Tensor dL_dopacity = grad_opacity.has_value() ? grad_opacity.value() : torch::zeros({P, 1}, extras.options());
  Tensor dL_dextras  = torch::zeros({P, E}, extras.options());
  if (P == 0) return std::make_tuple(dL_dextras, dL_dmeans2D, dL_dconic, dL_dopacity);

  char* geom_buffer         = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
  char* binning_buffer      = reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr());
  char* img_buffer          = reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr());
  GeometryState geomState   = GeometryState::fromChunk(geom_buffer, P);
  BinningState binningState = BinningState::fromChunk(binning_buffer, num_rendered);
  ImageState imgState       = ImageState::fromChunk(img_buffer, width * height);

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);
  render_extra_backward_kernel KERNEL_ARG(tile_grid, block)(width, height, E, imgState.ranges, binningState.point_list,
      geomState.means2D, geomState.conic_opacity, out_opacity.data_ptr<float>(), imgState.n_contrib,
      extras.contiguous().data_ptr<float>(), grad_pixel_extras.contiguous().data_ptr<float>(),
      (float3*) dL_dmeans2D.data_ptr<float>(), (float4*) dL_dconic.data_ptr<float>(), dL_dopacity.data_ptr<float>(),
      dL_dextras.data_ptr<float>());
  CHECK_CUDA_ERROR("render_extra_backward_kernel");
  return std::make_tuple(dL_dextras, dL_dmeans2D, dL_dconic, dL_dopacity);
}
REGIST_PYTORCH_EXTENSION(nerf_gaussian_rasterize_extra, {
  m.def("gaussian_rasterize_extra_forward", &render_extra_forward, "gaussian rasterize extra_forward (CUDA)");
  m.def("gaussian_rasterize_extra_backward", &render_extra_backward, "gaussian rasterize extra_backward (CUDA)");
})
}  // namespace GaussianRasterizer