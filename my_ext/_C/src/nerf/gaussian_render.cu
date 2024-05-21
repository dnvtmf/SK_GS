/*
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code:  https://github.com/graphdeco-inria/diff-gaussian-rasterization
*/
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "util.cuh"

namespace cg = cooperative_groups;
namespace GaussianRasterizer {

// Main rasterization method. Collaboratively works on one tile per block, each thread treats one pixel.
// Alternates between fetching and rasterizing data.
template <uint32_t CHANNELS, uint32_t NUM_EXTRA>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) renderCUDA_forward(const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list, int W, int H, const float2* __restrict__ points_xy_image,
    const float* __restrict__ features, const float4* __restrict__ conic_opacity, const float* extra,
    uint32_t* __restrict__ n_contrib, /*const float* __restrict__ bg_color,*/
    float* __restrict__ out_color, float* __restrict__ out_opacity, float* __restrict__ out_extra) {
  // Identify current tile and associated min/max pixel range.
  auto block                 = cg::this_thread_block();
  uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  uint32_t pix_id            = W * pix.y + pix.x;
  float2 pixf                = {(float) pix.x, (float) pix.y};

  // Check if this thread is associated with a valid pixel or outside.
  bool inside = pix.x < W && pix.y < H;
  // Done threads can help with fetching, but don't rasterize
  bool done = !inside;

  // Load start/end range of IDs to process in bit sorted list.
  uint2 range      = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int toDo         = range.y - range.x;

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

  // Initialize helper variables
  float T                       = 1.0f;
  uint32_t contributor          = 0;
  uint32_t last_contributor     = 0;
  float C[CHANNELS + NUM_EXTRA] = {0};
  float* E                      = C + CHANNELS;

  // Iterate over batches until all done or range is complete
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
    // End if entire block votes that it is done rasterizing
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

      // Eq. (3) from 3D Gaussian splatting paper.
      for (int ch = 0; ch < CHANNELS; ch++) C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
      if constexpr (NUM_EXTRA > 0)
        for (int ch = 0; ch < NUM_EXTRA; ch++) E[ch] += extra[collected_id[j] * NUM_EXTRA + ch] * alpha * T;

      T = test_T;

      // Keep track of last range entry to update this pixel.
      last_contributor = contributor;
    }
  }

  // All threads that treat valid pixel write out their final rendering data to the frame and auxiliary buffers.
  if (inside) {
    out_opacity[pix_id] = 1.f - T;
    n_contrib[pix_id]   = last_contributor;
    for (int ch = 0; ch < CHANNELS; ch++) out_color[ch * H * W + pix_id] = C[ch];  // + T * bg_color[ch];
    if constexpr (NUM_EXTRA > 0)
      for (int ch = 0; ch < NUM_EXTRA; ch++) out_extra[ch * H * W + pix_id] = E[ch];
  }
}

void render_forward(const dim3 grid, dim3 block, const uint2* ranges, const uint32_t* point_list, int W, int H, int E,
    const float2* means2D, const float* colors, const float4* conic_opacity, const float* extra, uint32_t* n_contrib,
    float* out_color, float* out_opacity, float* out_extra) {
  switch (E) {
    case 0:
      renderCUDA_forward<NUM_CHANNELS, 0> KERNEL_ARG(grid, block)(ranges, point_list, W, H, means2D, colors,
          conic_opacity, extra, n_contrib, out_color, out_opacity, out_extra);
      break;
    case 1:
      renderCUDA_forward<NUM_CHANNELS, 1> KERNEL_ARG(grid, block)(ranges, point_list, W, H, means2D, colors,
          conic_opacity, extra, n_contrib, out_color, out_opacity, out_extra);
      break;
    case 2:
      renderCUDA_forward<NUM_CHANNELS, 2> KERNEL_ARG(grid, block)(ranges, point_list, W, H, means2D, colors,
          conic_opacity, extra, n_contrib, out_color, out_opacity, out_extra);
      break;
    case 3:
      renderCUDA_forward<NUM_CHANNELS, 3> KERNEL_ARG(grid, block)(ranges, point_list, W, H, means2D, colors,
          conic_opacity, extra, n_contrib, out_color, out_opacity, out_extra);
      break;
    case 4:
      renderCUDA_forward<NUM_CHANNELS, 4> KERNEL_ARG(grid, block)(ranges, point_list, W, H, means2D, colors,
          conic_opacity, extra, n_contrib, out_color, out_opacity, out_extra);
      break;
    default: BCNN_ASSERT(false, "Only Support 0,1,2,3,4 extra features"); break;
  }
  CHECK_CUDA_ERROR("render_forward");
}

std::tuple<Tensor, Tensor, torch::optional<Tensor>> gaussian_rasterize_forward(int width, int height, int num_rendered,
    const Tensor& colors, const at::optional<Tensor>& extras, const Tensor& geomBuffer, const Tensor& binningBuffer,
    const Tensor& imageBuffer) {
  CHECK_INPUT(colors);
  CHECK_NDIM(colors, 2);
  int P = colors.size(0);  // num points
  int E = 0;               // num extras
  if (extras.has_value()) {
    CHECK_NDIM(extras.value(), 2);
    E = extras.value().size(1);  // num extras
    CHECK_INPUT(extras.value());
    CHECK_SHAPE(extras.value(), P, E);
  }
  Tensor pixel_colors  = torch::zeros({width, height, E}, colors.options());
  Tensor pixel_opacity = torch::zeros({width, height, E}, colors.options());
  at::optional<Tensor> pixel_extras;
  if (extras.has_value()) pixel_extras = torch::zeros({width, height, E}, colors.options());
  if (P == 0) return std::make_tuple(pixel_colors, pixel_opacity, pixel_extras);

  char* geom_buffer         = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
  char* binning_buffer      = reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr());
  char* img_buffer          = reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr());
  GeometryState geomState   = GeometryState::fromChunk(geom_buffer, P);
  BinningState binningState = BinningState::fromChunk(binning_buffer, num_rendered);
  ImageState imgState       = ImageState::fromChunk(img_buffer, width * height);

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  render_forward(tile_grid, block, imgState.ranges, binningState.point_list, width, height, E, geomState.means2D,
      colors.contiguous().data_ptr<float>(), geomState.conic_opacity,
      extras.has_value() ? extras.value().contiguous().data_ptr<float>() : nullptr, imgState.n_contrib,
      pixel_colors.data_ptr<float>(), pixel_opacity.data_ptr<float>(),
      extras.has_value() ? pixel_extras.value().data_ptr<float>() : nullptr);
  CHECK_CUDA_ERROR("render_forward");
  return std::make_tuple(pixel_colors, pixel_opacity, pixel_extras);
}

// Backward version of the rendering procedure.
template <uint32_t C, uint32_t E = 0>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) renderCUDA_backward(int P, int W, int H,
    const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list, /*const float* __restrict__ bg_color,*/
    const float2* __restrict__ points_xy_image, const float4* __restrict__ conic_opacity,
    const float* __restrict__ colors, const float* __restrict__ extras, const float* __restrict__ out_opacity,
    const uint32_t* __restrict__ n_contrib, const float* __restrict__ dL_dpixels,
    const float* __restrict__ dL_dout_extra, const float* __restrict__ dL_dout_opacity, float3* __restrict__ dL_dmean2D,
    float4* __restrict__ dL_dconic2D, float* __restrict__ dL_dopacity, float* __restrict__ dL_dcolors,
    float* __restrict__ dL_dextras) {
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

  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  bool done = !inside;
  int toDo  = range.y - range.x;

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
  __shared__ float collected_colors[(C + E) * BLOCK_SIZE];
  float* collected_extras = collected_colors + C * BLOCK_SIZE;

  // In the forward, we stored the final value for T, the product of all (1 - alpha) factors.
  const float T_final = inside ? 1.0f - out_opacity[pix_id] : 0;
  float T             = T_final;
  const float dL_dT   = -dL_dout_opacity[pix_id];

  // We start from the back. The ID of the last contributing Gaussian is known from each pixel from the forward.
  uint32_t contributor       = toDo;
  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  float accum_rec[C + E] = {0};
  float dL_dpixel[C + E];
  float* accum_ext = accum_rec + C;
  float* dL_doute  = dL_dpixel + C;

  if (inside) {
#pragma unroll
    for (int c = 0; c < C; c++) dL_dpixel[c] = dL_dpixels[c * H * W + pix_id];
#pragma unroll
    for (int e = 0; e < E; e++) dL_doute[e] = dL_dout_extra[e * H * W + pix_id];
  }
  float last_alpha        = 0;
  float last_color[C + E] = {0};
  float* last_extra       = last_color + C;

  // Gradient of pixel coordinate w.r.t. normalized screen-space viewport corrdinates (-1 to 1)
  const float ddelx_dx = 0.5 * W;
  const float ddely_dy = 0.5 * H;

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
      for (int c = 0; c < C; c++) collected_colors[c * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + c];
#pragma unroll
      for (int e = 0; e < E; e++) collected_extras[e * BLOCK_SIZE + block.thread_rank()] = extras[coll_id * E + e];
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

      const float dchannel_dcolor = alpha * T;

      // Propagate gradients to per-Gaussian colors and keep gradients w.r.t. alpha
      // (blending factor for a Gaussian/pixel pair).
      float dL_dalpha     = 0.0f;
      const int global_id = collected_id[j];
#pragma unroll
      for (int ch = 0; ch < C; ch++) {
        const float c = collected_colors[ch * BLOCK_SIZE + j];
        // Update last color (to be used in the next iteration)
        accum_rec[ch]  = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
        last_color[ch] = c;

        const float dL_dchannel = dL_dpixel[ch];
        dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
        // Update the gradients w.r.t. color of the Gaussian.
        // Atomic, since this pixel is just one of potentially many that were affected by this Gaussian.
        atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
      }
#pragma unroll
      for (int ch = 0; ch < E; ch++) {
        const float c = collected_extras[ch * BLOCK_SIZE + j];
        // Update last color (to be used in the next iteration)
        accum_ext[ch]  = last_alpha * last_extra[ch] + (1.f - last_alpha) * accum_ext[ch];
        last_extra[ch] = c;

        const float dL_dchannel = dL_doute[ch];
        dL_dalpha += (c - accum_ext[ch]) * dL_dchannel;
        // Update the gradients w.r.t. color of the Gaussian.
        // Atomic, since this pixel is just one of potentially many that were affected by this Gaussian.
        atomicAdd(&(dL_dextras[global_id * E + ch]), dchannel_dcolor * dL_dchannel);
      }
      dL_dalpha *= T;
      // Update last alpha (to be used in the next iteration)
      last_alpha = alpha;

      // Account for fact that alpha also influences how much of
      // the background color is added if nothing left to blend
      //   float bg_dot_dpixel = 0;
      //   for (int i = 0; i < C; i++) bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
      //   dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
      dL_dalpha += (-T_final / (1.f - alpha)) * dL_dT;

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

void render_backward(int P, int W, int H, int E, const dim3 grid, const dim3 block, const uint2* ranges,
    const uint32_t* point_list, /*const float* bg_color,*/ const float2* means2D, const float4* conic_opacity,
    const float* colors, const float* extras, const float* out_opacity, const uint32_t* n_contrib,
    const float* dL_dpixels, const float* dL_dout_extras, const float* dL_dout_opacity, float3* dL_dmean2D,
    float4* dL_dconic2D, float* dL_dopacity, float* dL_dcolors, float* dL_dextras) {
  switch (E) {
    case 0:
      renderCUDA_backward<NUM_CHANNELS, 0> KERNEL_ARG(grid, block)(P, W, H, ranges, point_list, means2D, conic_opacity,
          colors, extras, out_opacity, n_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity, dL_dmean2D, dL_dconic2D,
          dL_dopacity, dL_dcolors, dL_dextras);
      break;
    case 1:
      renderCUDA_backward<NUM_CHANNELS, 1> KERNEL_ARG(grid, block)(P, W, H, ranges, point_list, means2D, conic_opacity,
          colors, extras, out_opacity, n_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity, dL_dmean2D, dL_dconic2D,
          dL_dopacity, dL_dcolors, dL_dextras);
      break;
    case 2:
      renderCUDA_backward<NUM_CHANNELS, 2> KERNEL_ARG(grid, block)(P, W, H, ranges, point_list, means2D, conic_opacity,
          colors, extras, out_opacity, n_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity, dL_dmean2D, dL_dconic2D,
          dL_dopacity, dL_dcolors, dL_dextras);
      break;
    case 3:
      renderCUDA_backward<NUM_CHANNELS, 3> KERNEL_ARG(grid, block)(P, W, H, ranges, point_list, means2D, conic_opacity,
          colors, extras, out_opacity, n_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity, dL_dmean2D, dL_dconic2D,
          dL_dopacity, dL_dcolors, dL_dextras);
      break;
    case 4:
      renderCUDA_backward<NUM_CHANNELS, 4> KERNEL_ARG(grid, block)(P, W, H, ranges, point_list, means2D, conic_opacity,
          colors, extras, out_opacity, n_contrib, dL_dpixels, dL_dout_extras, dL_dout_opacity, dL_dmean2D, dL_dconic2D,
          dL_dopacity, dL_dcolors, dL_dextras);
      break;
    default: BCNN_ASSERT(false, "Only support NUM_EXTRA=0,1,2,3,4"); break;
  }
}

std::tuple<Tensor, at::optional<Tensor>, Tensor, Tensor, Tensor> gaussian_rasterize_backward(int weight, int height,
    int P, const at::optional<Tensor>& colors, const at::optional<Tensor> extras, const Tensor& out_opacity,
    const torch::Tensor& dL_dout_color, const Tensor& dL_dout_opacity, const at::optional<Tensor>& dL_dout_extra,
    const int R, const torch::Tensor& geomBuffer, const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer) {
  // const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  const int E = (extras.has_value() && dL_dout_extra.has_value()) ? extras.value().size(-1) : 0;

  auto options              = out_opacity.options();
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, options);
  torch::Tensor dL_dconic   = torch::zeros({P, 2, 2}, options);
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, options);
  torch::Tensor dL_dcolors  = torch::zeros({P, NUM_CHANNELS}, options);
  at::optional<Tensor> dL_dextras;
  if (E > 0) dL_dextras = torch::zeros({P, E}, options);

  if (P == 0) return std::make_tuple(dL_dcolors, dL_dextras, dL_dmeans2D, dL_dconic, dL_dopacity);

  char* geom_buffer         = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
  char* binning_buffer      = reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr());
  char* img_buffer          = reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr());
  GeometryState geomState   = GeometryState::fromChunk(geom_buffer, P);
  BinningState binningState = BinningState::fromChunk(binning_buffer, R);
  ImageState imgState       = ImageState::fromChunk(img_buffer, W * H);

  const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  const float* color_ptr = colors.has_value() ? colors.value().contiguous().data<float>() : geomState.rgb;

  render_backward(P, W, H, E, tile_grid, block, imgState.ranges, binningState.point_list, geomState.means2D,
      geomState.conic_opacity, color_ptr, E > 0 ? extras.value().contiguous().data<float>() : nullptr,
      out_opacity.contiguous().data<float>(), imgState.n_contrib, dL_dout_color.contiguous().data<float>(),
      E > 0 ? dL_dout_extra.value().contiguous().data<float>() : nullptr, dL_dout_opacity.contiguous().data<float>(),
      (float3*) dL_dmeans2D.contiguous().data<float>(), (float4*) dL_dconic.contiguous().data<float>(),
      dL_dopacity.contiguous().data<float>(), dL_dcolors.contiguous().data<float>(),
      E > 0 ? dL_dextras.value().data<float>() : nullptr);
  CHECK_CUDA_ERROR("render");
  return std::make_tuple(dL_dcolors, dL_dextras, dL_dmeans2D, dL_dconic, dL_dopacity);
}

REGIST_PYTORCH_EXTENSION(nerf_gaussian_render, {
  m.def("gaussian_rasterize_forward", &gaussian_rasterize_forward, "gaussian_rasterize_forward (CUDA)");
  m.def("gaussian_rasterize_backward", &gaussian_rasterize_backward, "gaussian_rasterize_backward (CUDA)");
})
}  // namespace GaussianRasterizer