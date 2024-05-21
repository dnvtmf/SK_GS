#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "gaussian_render.h"
#include "util.cuh"

namespace cg = cooperative_groups;
namespace GaussianRasterizer {

__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    render_topk_weights(const int topk, const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
        int W, int H, const float2* __restrict__ points_xy_image, const float4* __restrict__ conic_opacity,
        const uint32_t* __restrict__ n_contrib, int32_t* __restrict__ top_indices, float* __restrict__ top_weights) {
  // Identify current tile and associated min/max pixel range.
  auto block                       = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
  const uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  const uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
  const uint32_t pix_id            = W * pix.y + pix.x;
  const float2 pixf                = {(float) pix.x, (float) pix.y};

  top_weights += pix_id * topk;
  top_indices += pix_id * topk;

  const bool inside = pix.x < W && pix.y < H;
  const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  bool done = !inside;
  int toDo  = range.y - range.x;

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

  // Initialize helper variables
  float T                    = 1.0f;
  uint32_t contributor       = 0;
  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  // Iterate over batches until all done or range is complete
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
    block.sync();
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
      if (contributor >= last_contributor) continue;

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

      float w = alpha * T;
      int idx = collected_id[j];
      for (int k = 0; k < topk; ++k) {
        if (w >= top_weights[k]) {
          auto tw        = top_weights[k];
          top_weights[k] = w;
          w              = tw;
          auto t_idx     = top_indices[k];
          top_indices[k] = idx;
          idx            = t_idx;
        }
      }
      T = test_T;
    }
  }
}

std::tuple<Tensor, Tensor> gaussian_topk_weights(int topk, int W, int H, int P, const int R,
    const torch::Tensor& geomBuffer, const torch::Tensor& binningBuffer, const torch::Tensor& imageBuffer) {
  char* geom_buffer         = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
  char* binning_buffer      = reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr());
  char* img_buffer          = reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr());
  GeometryState geomState   = GeometryState::fromChunk(geom_buffer, P);
  BinningState binningState = BinningState::fromChunk(binning_buffer, R);
  ImageState imgState       = ImageState::fromChunk(img_buffer, W * H);

  Tensor top_indices = torch::full({H, W, topk}, -1, geomBuffer.options().dtype(torch::kInt32));
  Tensor top_weights = torch::full({H, W, topk}, 0, geomBuffer.options().dtype(torch::kFloat32));
  if (P == 0) return std::make_tuple(top_indices, top_weights);

  const dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);
  render_topk_weights KERNEL_ARG(tile_grid, block)(topk, imgState.ranges, binningState.point_list, W, H,
      geomState.means2D, geomState.conic_opacity, imgState.n_contrib, top_indices.data<int>(),
      top_weights.data<float>());
  CHECK_CUDA_ERROR("render_topk_weights");

  return std::make_tuple(top_indices, top_weights);
}

REGIST_PYTORCH_EXTENSION(nerf_gaussian_topk, { m.def("gaussian_topk_weights", &gaussian_topk_weights); })
}  // namespace GaussianRasterizer