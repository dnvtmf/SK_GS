#include <algorithm>
#include <cmath>
#include <vector>

#include "common.hpp"

std::vector<Tensor> iterative_normalization_forward(Tensor input, Tensor running_mean, Tensor running_wm, int nc,
    int nIter, double eps, double momentum, bool training) {
  BCNN_ASSERT(input.size(1) % nc == 0, "channels must be multiply of nc.");
  int ng = input.size(1) / nc;
  int m  = input.numel() / input.size(1);

  std::vector<Tensor> output(training ? nIter + 4 : 2);

  Tensor& x  = output[training ? nIter + 0 : 0];
  Tensor& xn = output[training ? nIter + 3 : 1];

  x = input.transpose(0, 1).contiguous().view({ng, nc, m});
  if (training) {
    auto& sn  = output[nIter + 1];
    auto& rtr = output[nIter + 2];
    auto mean = x.mean(/*dim=*/2, /*keepdim=*/true);
    auto I    = at::eye(nc, input.options()).expand({ng, nc, nc});
    x.sub_(mean);
    sn  = at::baddbmm(I, x, x.transpose(1, 2), eps, 1. / m);
    rtr = sn.mul(I).sum({1, 2}, true).reciprocal_();
    sn.mul_(rtr);
    output[0] = 1.5 * I - 0.5 * sn;
    for (int i = 1; i < nIter; ++i)
      output[i] = at::baddbmm(output[i - 1], at::matrix_power(output[i - 1], 3), sn, 1.5, -0.5);
    output[nIter - 1].mul_(rtr.sqrt());
    running_mean.mul_(1. - momentum).add_(mean * momentum);
    running_wm.mul_(1. - momentum).add_(output[nIter - 1] * momentum);
    xn = output[nIter - 1].matmul(x);
  } else {
    x.sub_(running_mean);
    xn = running_wm.matmul(x);
  }
  auto size = input.sizes().vec();
  std::swap(size[0], size[1]);
  xn = xn.resize_(size).transpose_(0, 1).contiguous();
  return output;
}

Tensor iterative_normalization_backward(Tensor grad, std::vector<Tensor> saved) {
  int nIter = saved.size() - 3;
  auto& wm  = saved[nIter - 1];
  auto& xc  = saved[nIter + 0];
  auto& sn  = saved[nIter + 1];
  auto& rtr = saved[nIter + 2];
  // int ng    = xc.size(0);
  int nc = xc.size(1);
  int m  = xc.size(2);

  BCNN_ASSERT(nIter >= 0, "Error saved variables for backward.");

  sn.transpose_(1, 2);
  auto g    = grad.transpose(0, 1).contiguous().view_as(xc);
  auto g_wm = g.matmul(xc.transpose(1, 2));
  auto g_P  = g_wm.mul(rtr.sqrt());
  auto g_sn = at::zeros_like(g_P);

  for (int k = nIter - 2; k >= 0; --k) {
    auto& P = saved[k].transpose_(1, 2);
    auto P2 = P.matmul(P);
    g_sn.add_(P2.matmul(P).matmul(g_P));
    auto tmp = g_P.matmul(sn);
    g_P.baddbmm_(tmp, P2, 1.5, -0.5);
    g_P.baddbmm_(P2, tmp, 1, -0.5);
    g_P.baddbmm_(P.matmul(tmp), P, 1., -0.5);
  }
  g_sn.add_(g_P);
  auto I    = at::eye(nc, xc.options());
  auto g_tr = g_wm.transpose(1, 2).matmul(wm).sub_(sn.matmul(g_sn)).mul_(I).sum({1, 2}, true).mul(I);
  g_sn.add_(g_sn.transpose(1, 2)).add_(g_tr.mul_(2.)).mul_(rtr.mul_(-0.5 / m));
  auto g_x  = at::baddbmm(wm.matmul(g - g.mean(-1, true)), g_sn, xc);
  auto size = grad.sizes().vec();
  std::swap(size[0], size[1]);
  return g_x.resize_(size).transpose_(0, 1).contiguous();
}

REGIST_PYTORCH_EXTENSION(iterative_normalization, {
  m.def("iterative_normalization_forward", &iterative_normalization_forward, "Iterative Normalization Forward");
  m.def("iterative_normalization_backward", &iterative_normalization_backward, "Iterative Normalization Backward");
})