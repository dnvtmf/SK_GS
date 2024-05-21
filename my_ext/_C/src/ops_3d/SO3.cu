#include "ops_3d.h"
#include "util.cuh"
#include "glm_helper.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

namespace OPS_3D {
Tensor rotate(vector<Tensor> x, std::string order) {
  auto shape = x[0].sizes().vec();
  shape.push_back(3);
  shape.push_back(3);
  Tensor R = torch::zeros(shape, x[0].options());
  return R;
}
}  // namespace OPS_3D
