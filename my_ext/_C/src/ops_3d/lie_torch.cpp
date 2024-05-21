/*
reference: https://github.com/princeton-vl/lietorch/
*/

#include "common.hpp"
#include "lie.h"

namespace Lie {

// unary operations
torch::Tensor exp_forward_cpu(int, torch::Tensor);
std::vector<torch::Tensor> exp_backward_cpu(int, torch::Tensor, torch::Tensor);

torch::Tensor log_forward_cpu(int, torch::Tensor);
std::vector<torch::Tensor> log_backward_cpu(int, torch::Tensor, torch::Tensor);

torch::Tensor inv_forward_cpu(int, torch::Tensor);
std::vector<torch::Tensor> inv_backward_cpu(int, torch::Tensor, torch::Tensor);

// binary operations
torch::Tensor mul_forward_cpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> mul_backward_cpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor adj_forward_cpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> adj_backward_cpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor adjT_forward_cpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> adjT_backward_cpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor act_forward_cpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> act_backward_cpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor act4_forward_cpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> act4_backward_cpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

// conversion operations
// std::vector<torch::Tensor> to_vec_backward_cpu(int, torch::Tensor, torch::Tensor);
// std::vector<torch::Tensor> from_vec_backward_cpu(int, torch::Tensor, torch::Tensor);

// utility operations
torch::Tensor orthogonal_projector_cpu(int, torch::Tensor);

torch::Tensor as_matrix_forward_cpu(int, torch::Tensor);

torch::Tensor jleft_forward_cpu(int, torch::Tensor, torch::Tensor);

// unary operations
torch::Tensor exp_forward_gpu(int, torch::Tensor);
std::vector<torch::Tensor> exp_backward_gpu(int, torch::Tensor, torch::Tensor);

torch::Tensor log_forward_gpu(int, torch::Tensor);
std::vector<torch::Tensor> log_backward_gpu(int, torch::Tensor, torch::Tensor);

torch::Tensor inv_forward_gpu(int, torch::Tensor);
std::vector<torch::Tensor> inv_backward_gpu(int, torch::Tensor, torch::Tensor);

// binary operations
torch::Tensor mul_forward_gpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> mul_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor adj_forward_gpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> adj_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor adjT_forward_gpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> adjT_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor act_forward_gpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> act_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor act4_forward_gpu(int, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> act4_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor);

// conversion operations
// std::vector<torch::Tensor> to_vec_backward_gpu(int, torch::Tensor, torch::Tensor);
// std::vector<torch::Tensor> from_vec_backward_gpu(int, torch::Tensor, torch::Tensor);

// utility operators
torch::Tensor orthogonal_projector_gpu(int, torch::Tensor);

torch::Tensor as_matrix_forward_gpu(int, torch::Tensor);

torch::Tensor jleft_forward_gpu(int, torch::Tensor, torch::Tensor);

/* Interface for cuda and c++ group operations

    enum group_t { SO3=1, SE3=2, Sim3=3 };
    X, Y, Z: (uppercase) Lie Group Elements
    a, b, c: (lowercase) Lie Algebra Elements
*/

// Unary operations
torch::Tensor expm(int group_index, torch::Tensor a) {
  CHECK_CONTIGUOUS(a);
  if (a.device().type() == torch::DeviceType::CPU) {
    return exp_forward_cpu(group_index, a);

  } else if (a.device().type() == torch::DeviceType::CUDA) {
    return exp_forward_gpu(group_index, a);
  }

  return a;
}

std::vector<torch::Tensor> expm_backward(int group_index, torch::Tensor grad, torch::Tensor a) {
  CHECK_CONTIGUOUS(a);
  CHECK_CONTIGUOUS(grad);
  if (a.device().type() == torch::DeviceType::CPU) {
    return exp_backward_cpu(group_index, grad, a);

  } else if (a.device().type() == torch::DeviceType::CUDA) {
    return exp_backward_gpu(group_index, grad, a);
  }

  return {};
}

torch::Tensor logm(int group_index, torch::Tensor X) {
  CHECK_CONTIGUOUS(X);
  if (X.device().type() == torch::DeviceType::CPU) {
    return log_forward_cpu(group_index, X);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return log_forward_gpu(group_index, X);
  }

  return X;
}

std::vector<torch::Tensor> logm_backward(int group_index, torch::Tensor grad, torch::Tensor X) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(grad);

  if (X.device().type() == torch::DeviceType::CPU) {
    return log_backward_cpu(group_index, grad, X);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return log_backward_gpu(group_index, grad, X);
  }

  return {};
}

torch::Tensor inv(int group_index, torch::Tensor X) {
  CHECK_CONTIGUOUS(X);

  if (X.device().type() == torch::DeviceType::CPU) {
    return inv_forward_cpu(group_index, X);
  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return inv_forward_gpu(group_index, X);
  }

  return X;
}

std::vector<torch::Tensor> inv_backward(int group_index, torch::Tensor grad, torch::Tensor X) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(grad);

  if (X.device().type() == torch::DeviceType::CPU) {
    return inv_backward_cpu(group_index, grad, X);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return inv_backward_gpu(group_index, grad, X);
  }

  return {};
}

// Binary operations

torch::Tensor mul(int group_index, torch::Tensor X, torch::Tensor Y) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(Y);

  if (X.device().type() == torch::DeviceType::CPU) {
    return mul_forward_cpu(group_index, X, Y);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return mul_forward_gpu(group_index, X, Y);
  }

  return X;
}

std::vector<torch::Tensor> mul_backward(int group_index, torch::Tensor grad, torch::Tensor X, torch::Tensor Y) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(Y);
  CHECK_CONTIGUOUS(grad);

  if (X.device().type() == torch::DeviceType::CPU) {
    return mul_backward_cpu(group_index, grad, X, Y);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return mul_backward_gpu(group_index, grad, X, Y);
  }

  return {};
}

torch::Tensor adj(int group_index, torch::Tensor X, torch::Tensor a) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(a);

  if (X.device().type() == torch::DeviceType::CPU) {
    return adj_forward_cpu(group_index, X, a);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return adj_forward_gpu(group_index, X, a);
  }

  return X;
}

std::vector<torch::Tensor> adj_backward(int group_index, torch::Tensor grad, torch::Tensor X, torch::Tensor a) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(a);
  CHECK_CONTIGUOUS(grad);

  if (X.device().type() == torch::DeviceType::CPU) {
    return adj_backward_cpu(group_index, grad, X, a);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return adj_backward_gpu(group_index, grad, X, a);
  }

  return {};
}

torch::Tensor adjT(int group_index, torch::Tensor X, torch::Tensor a) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(a);

  if (X.device().type() == torch::DeviceType::CPU) {
    return adjT_forward_cpu(group_index, X, a);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return adjT_forward_gpu(group_index, X, a);
  }

  return X;
}

std::vector<torch::Tensor> adjT_backward(int group_index, torch::Tensor grad, torch::Tensor X, torch::Tensor a) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(a);
  CHECK_CONTIGUOUS(grad);

  if (X.device().type() == torch::DeviceType::CPU) {
    return adjT_backward_cpu(group_index, grad, X, a);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return adjT_backward_gpu(group_index, grad, X, a);
  }

  return {};
}

torch::Tensor act(int group_index, torch::Tensor X, torch::Tensor p) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(p);

  if (X.device().type() == torch::DeviceType::CPU) {
    return act_forward_cpu(group_index, X, p);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return act_forward_gpu(group_index, X, p);
  }

  return X;
}

std::vector<torch::Tensor> act_backward(int group_index, torch::Tensor grad, torch::Tensor X, torch::Tensor p) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(p);
  CHECK_CONTIGUOUS(grad);

  if (X.device().type() == torch::DeviceType::CPU) {
    return act_backward_cpu(group_index, grad, X, p);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return act_backward_gpu(group_index, grad, X, p);
  }

  return {};
}

torch::Tensor act4(int group_index, torch::Tensor X, torch::Tensor p) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(p);

  if (X.device().type() == torch::DeviceType::CPU) {
    return act4_forward_cpu(group_index, X, p);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return act4_forward_gpu(group_index, X, p);
  }

  return X;
}

std::vector<torch::Tensor> act4_backward(int group_index, torch::Tensor grad, torch::Tensor X, torch::Tensor p) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(p);
  CHECK_CONTIGUOUS(grad);

  if (X.device().type() == torch::DeviceType::CPU) {
    return act4_backward_cpu(group_index, grad, X, p);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return act4_backward_gpu(group_index, grad, X, p);
  }

  return {};
}

torch::Tensor projector(int group_index, torch::Tensor X) {
  CHECK_CONTIGUOUS(X);

  if (X.device().type() == torch::DeviceType::CPU) {
    return orthogonal_projector_cpu(group_index, X);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return orthogonal_projector_gpu(group_index, X);
  }

  return X;
}

torch::Tensor as_matrix(int group_index, torch::Tensor X) {
  CHECK_CONTIGUOUS(X);

  if (X.device().type() == torch::DeviceType::CPU) {
    return as_matrix_forward_cpu(group_index, X);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return as_matrix_forward_gpu(group_index, X);
  }

  return X;
}

torch::Tensor Jinv(int group_index, torch::Tensor X, torch::Tensor a) {
  CHECK_CONTIGUOUS(X);
  CHECK_CONTIGUOUS(a);

  if (X.device().type() == torch::DeviceType::CPU) {
    return jleft_forward_cpu(group_index, X, a);

  } else if (X.device().type() == torch::DeviceType::CUDA) {
    return jleft_forward_gpu(group_index, X, a);
  }

  return a;
}

// {exp, log, inv, mul, adj, adjT, act, act4} forward/backward bindings
REGIST_PYTORCH_EXTENSION(ops_3d_lie, {
  m.def("lie_expm", &expm, "exp map forward");
  m.def("lie_expm_backward", &expm_backward, "exp map backward");

  m.def("lie_logm", &logm, "log map forward");
  m.def("lie_logm_backward", &logm_backward, "log map backward");

  m.def("lie_inv", &inv, "inverse operator");
  m.def("lie_inv_backward", &inv_backward, "inverse operator backward");

  m.def("lie_mul", &mul, "group operator");
  m.def("lie_mul_backward", &mul_backward, "group operator backward");

  m.def("lie_adj", &adj, "adjoint operator");
  m.def("lie_adj_backward", &adj_backward, "adjoint operator backward");

  m.def("lie_adjT", &adjT, "transposed adjoint operator");
  m.def("lie_adjT_backward", &adjT_backward, "transposed adjoint operator backward");

  m.def("lie_act", &act, "action on point");
  m.def("lie_act_backward", &act_backward, "action on point backward");

  m.def("lie_act4", &act4, "action on homogeneous point");
  m.def("lie_act4_backward", &act4_backward, "action on homogeneous point backward");

  // functions with no gradient
  m.def("lie_as_matrix", &as_matrix, "convert to matrix");
  m.def("lie_projector", &projector, "orthogonal projection matrix");
  m.def("lie_Jinv", &Jinv, "left inverse jacobian operator");
});

}  // namespace Lie
