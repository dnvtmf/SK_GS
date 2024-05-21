#include "util.cuh"

template <typename T, int op = 0>
void __global__ kernel_op_previous_forward(const uint32_t M, const uint32_t B, const T *__restrict__ x,
    const int *__restrict__ pack_info, const int first_type, T first_value, const T *__restrict__ first_tensor,
    T *output) {
  const uint32_t n = blockIdx.x;
  if (n >= B) return;
  const uint32_t offset = pack_info ? pack_info[n * 2 + 0] : n * (M / B);
  const uint32_t N      = pack_info ? pack_info[n * 2 + 1] : M / B;
  //   for (int c = 0; c < C; ++c) {
  x += offset;
  output += offset;
  for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
    T last, now = x[i];
    if (i == 0) {
      if (first_type == 0)
        last = x[i];
      else if (first_type == 1)
        last = first_value;
      else
        last = first_tensor[n];
    } else {
      last = x[i - 1];
    }
    if (op == 0)
      output[i] = last;
    else if (op == 1)
      output[i] = now + last;
    else if (op == 2)
      output[i] = now - last;
    else if (op == 3)
      output[i] = now * last;
    else if (op == 4)
      output[i] = min(now, last);
    else if (op == 5)
      output[i] = max(now, last);
    else if (op == 6)
      output[i] = 0.5 * (now + last);
  }
  //   }
}

#define EXPAND_OP_CASE(OP)                                                                                     \
  kernel_op_previous_forward<scalar_t, OP> KERNEL_ARG(B, 256)(M, B, value.data_ptr<scalar_t>(),                \
      pack_info.has_value() ? pack_info.value().data_ptr<int>() : nullptr, first_type, (scalar_t) first_value, \
      first_tensor.has_value() ? first_tensor.value().data_ptr<scalar_t>() : nullptr, result.data_ptr<scalar_t>());

Tensor op_previous_forward(Tensor value, int op, at::optional<Tensor> pack_info, int first_type,
    at::optional<Tensor> first_tensor = at::nullopt, float first_value = 0) {
  value = value.contiguous();
  uint32_t M, B, C;
  if (pack_info.has_value()) {
    CHECK_TYPE(pack_info.value(), torch::kInt32);
    BCNN_ASSERT(pack_info.value().device() == value.device(), "Error device for pack_info");
    BCNN_ASSERT(value.ndimension() == 1 || value.ndimension() == 2, "Error shape of value");
    BCNN_ASSERT(pack_info.value().ndimension() == 2 && pack_info.value().size(1) == 2, "Error shape of pack_info");
    M = pack_info.value()[-1][0].item<int>();
    B = pack_info.value().size(0) - 1;
    C = value.numel() / M;
  } else {
    BCNN_ASSERT(value.ndimension() == 2 || value.ndimension() == 3, "Error shape of value");
    C = value.ndimension() == 2 ? 1 : value.size(2);
    B = value.size(0);
    M = value.numel() / C;
  }
  BCNN_ASSERT(0 <= first_type && first_type < 3, "first_type must be 0, 1, 2");
  if (first_type == 2) {
    BCNN_ASSERT(first_tensor.has_value(), "first_tensor must be not None when first type = 2");
    CHECK_CONTIGUOUS(first_tensor.value());
    BCNN_ASSERT(first_tensor.value().device() == value.device(), "Error device for first_tensor");
    BCNN_ASSERT(first_tensor.value().numel() == B * C, "error shape for first_tensor");
  }
  BCNN_ASSERT(0 <= op && op < 7, "Only support 0: set, 1: +, 2: -, 3: *, 4: min, 5: max, 6 mean");

  Tensor result = torch::zeros_like(value);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type(), "op_previous_forward", ([&] {
    if (value.is_cuda()) {
      BCNN_ASSERT(C == 1, "CUDA only support C == 1");
      switch (op) {
        case 0: EXPAND_OP_CASE(0); break;
        case 1: EXPAND_OP_CASE(1); break;
        case 2: EXPAND_OP_CASE(2); break;
        case 3: EXPAND_OP_CASE(3); break;
        case 4: EXPAND_OP_CASE(4); break;
        case 5: EXPAND_OP_CASE(5); break;
        case 6: EXPAND_OP_CASE(6); break;
        default: break;
      }
    } else {
      const scalar_t *x_ptr = value.data_ptr<scalar_t>();
      int *pack_p           = pack_info.has_value() ? pack_info.value().data_ptr<int>() : nullptr;
      scalar_t *f_ptr       = first_tensor.has_value() ? first_tensor.value().data_ptr<scalar_t>() : nullptr;
      scalar_t *o_ptr       = result.data_ptr<scalar_t>();
      for (uint32_t i = 0; i < B; ++i) {
        uint32_t offset = pack_p ? pack_p[i * 2 + 0] : i * (M / B);
        uint32_t num    = pack_p ? pack_p[i * 2 + 1] : (M / B);
        for (uint32_t c = 0; c < C; ++c) {
          scalar_t last;
          if (first_type == 0)
            last = x_ptr[offset * C + c];
          else if (first_type == 1)
            last = first_value;
          else
            last = f_ptr[i * C + c];
          for (uint32_t j = 0; j < num; ++j) {
            scalar_t now = x_ptr[(offset + j) * C + c], res = 0;
            switch (op) {
              case 0: res = last; break;
              case 1: res = now + last; break;
              case 2: res = now - last; break;
              case 3: res = now * last; break;
              case 4: res = min(now, last); break;
              case 5: res = max(now, last); break;
              case 6: res = 0.5 * (now + last); break;
              default: break;
            }
            o_ptr[(offset + j) * C + c] = res;
            last                        = now;
          }
        }
      }
    }
  }));
  return result;
}

#undef EXPAND_OP_CASE

template <typename T, int op = 0>
void __global__ kernel_op_previous_backward(const uint32_t M, const uint32_t B, const T *__restrict__ grad_o,
    const T *__restrict__ x, const int *__restrict__ pack_info, int first_type, T first_value,
    const T *__restrict__ first_tensor, T *grad_x, T *grad_first) {
  const uint32_t n = blockIdx.x;
  if (n >= B) return;
  const uint32_t offset = pack_info ? pack_info[n * 2 + 0] : n * (M / B);
  const uint32_t N      = pack_info ? pack_info[n * 2 + 1] : M / B;
  //   for (int c = 0; c < C; ++c) {
  x += offset;
  grad_o += offset;

  if (threadIdx.x == 0 && first_type == 2 && grad_first != nullptr) {
    if (op == 0) grad_first[n] = grad_o[0];
    if (op == 1) grad_first[n] = grad_o[0];
    if (op == 2) grad_first[n] = -grad_o[0];
    if (op == 3) grad_first[n] = grad_o[0] * x[0];
    if (op == 4) grad_first[n] = first_tensor[n] < x[0] ? grad_o[0] : T(0);
    if (op == 5) grad_first[n] = first_tensor[n] > x[0] ? grad_o[0] : T(0);
    if (op == 6) grad_first[n] = T(0.5) * grad_o[0];
  }
  if (grad_x == nullptr) return;

  grad_x += offset;
  for (uint32_t i = threadIdx.x; i < N; i += blockDim.x) {
    if (i == 0) {
      if (first_type == 0) {
        if (op == 0) grad_x[i] = grad_o[i] + (N > 1 ? grad_o[i + 1] : T(0));
        if (op == 1) grad_x[i] = (T) 2 * grad_o[i] + (N > 1 ? grad_o[i + 1] : T(0));
        if (op == 2) grad_x[i] = -(N > 1 ? grad_o[i + 1] : T(0));
        if (op == 3) grad_x[i] = (T) 2 * x[i] * grad_o[i] + (N > 1 ? grad_o[i + 1] * x[i + 1] : T(0));
        if (op == 4) grad_x[i] = grad_o[i] + (N > 1 && x[i] < x[i + 1] ? grad_o[i + 1] : T(0));
        if (op == 5) grad_x[i] = grad_o[i] + (N > 1 && x[i] > x[i + 1] ? grad_o[i + 1] : T(0));
        if (op == 6) grad_x[i] = grad_o[i] + (N > 1 ? T(0.5) * grad_o[i + 1] : T(0));
      } else {
        T last = (first_type == 1 ? first_value : first_tensor[n]);
        if (op == 0) grad_x[i] = (N > 1 ? grad_o[i + 1] : T(0));
        if (op == 1) grad_x[i] = grad_o[i] + (N > 1 ? grad_o[i + 1] : T(0));
        if (op == 2) grad_x[i] = grad_o[i] - (N > 1 ? grad_o[i + 1] : T(0));
        if (op == 3) grad_x[i] = grad_o[i] * last + (N > 1 ? grad_o[i + 1] * x[i + 1] : T(0));
        if (op == 4) grad_x[i] = (last < x[i] ? T(0) : grad_o[i]) + (N > 1 && x[i] < x[i + 1] ? grad_o[i + 1] : T(0));
        if (op == 5) grad_x[i] = (last > x[i] ? T(0) : grad_o[i]) + (N > 1 && x[i] > x[i + 1] ? grad_o[i + 1] : T(0));
        if (op == 6) grad_x[i] = T(0.5) * (grad_o[i] + (N > 1 ? grad_o[i + 1] : T(0)));
      }
    } else if (i + 1 == N) {
      if (op == 0) grad_x[i] = 0;
      if (op == 1) grad_x[i] = grad_o[i];
      if (op == 2) grad_x[i] = grad_o[i];
      if (op == 3) grad_x[i] = grad_o[i] * x[i - 1];
      if (op == 4) grad_x[i] = x[i - 1] < x[i] ? T(0) : grad_o[i];
      if (op == 5) grad_x[i] = x[i - 1] > x[i] ? T(0) : grad_o[i];
      if (op == 6) grad_x[i] = T(0.5) * grad_o[i];
    } else {
      if (op == 0) grad_x[i] = grad_o[i + 1];
      if (op == 1) grad_x[i] = grad_o[i] + grad_o[i + 1];
      if (op == 2) grad_x[i] = grad_o[i] - grad_o[i + 1];
      if (op == 3) grad_x[i] = grad_o[i] * x[i - 1] + grad_o[i + 1] * x[i + 1];
      if (op == 4) grad_x[i] = (x[i - 1] < x[i] ? T(0) : grad_o[i]) + (x[i] < x[i + 1] ? grad_o[i + 1] : T(0));
      if (op == 5) grad_x[i] = (x[i - 1] > x[i] ? T(0) : grad_o[i]) + (x[i] > x[i + 1] ? grad_o[i + 1] : T(0));
      if (op == 6) grad_x[i] = (grad_o[i] + grad_o[i + 1]) * T(0.5);
    }
  }
  //   }
}

#define EXPAND_OP_PREVIOUS_BACKWRAD_CASE(OP)                    \
  kernel_op_previous_backward<scalar_t, OP> KERNEL_ARG(B, 256)( \
      M, B, grad_o_ptr, x_ptr, pack_ptr, first_type, first_value, first_ptr, grad_x_ptr, grad_f_ptr);

vector<at::optional<Tensor>> op_previous_backward(Tensor grad_out, Tensor value, int op, at::optional<Tensor> pack_info,
    at::optional<Tensor> first_tensor, int first_type, float first_value, bool need_grad_value, bool need_grad_first) {
  grad_out = grad_out.contiguous();
  uint32_t M, B, C;
  if (pack_info.has_value()) {
    M = pack_info.value()[-1][0].item<int>();
    B = pack_info.value().size(0) - 1;
    C = value.numel() / M;
  } else {
    BCNN_ASSERT(value.ndimension() == 2 || value.ndimension() == 3, "Error shape of value");
    C = value.ndimension() == 2 ? 1 : value.size(2);
    B = value.size(0);
    M = value.numel() / C;
  }
  at::optional<Tensor> grad_value, grad_first;
  if (need_grad_value) grad_value = torch::zeros_like(value);
  if (need_grad_first) grad_first = torch::zeros_like(first_tensor.value());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(value.scalar_type(), "op_previous_backward", ([&] {
    scalar_t *grad_o_ptr = grad_out.data_ptr<scalar_t>();
    scalar_t *x_ptr      = value.data_ptr<scalar_t>();
    int *pack_ptr        = pack_info.has_value() ? pack_info.value().data_ptr<int>() : nullptr;
    scalar_t *first_ptr  = first_tensor.has_value() ? first_tensor.value().data_ptr<scalar_t>() : nullptr;
    scalar_t *grad_x_ptr = need_grad_value ? grad_value.value().data_ptr<scalar_t>() : nullptr;
    scalar_t *grad_f_ptr = need_grad_first ? grad_first.value().data_ptr<scalar_t>() : nullptr;

    if (value.is_cuda()) {
      BCNN_ASSERT(C == 1, "CUDA only support C == 1");
      switch (op) {
        case 0: EXPAND_OP_PREVIOUS_BACKWRAD_CASE(0); break;
        case 1: EXPAND_OP_PREVIOUS_BACKWRAD_CASE(1); break;
        case 2: EXPAND_OP_PREVIOUS_BACKWRAD_CASE(2); break;
        case 3: EXPAND_OP_PREVIOUS_BACKWRAD_CASE(3); break;
        case 4: EXPAND_OP_PREVIOUS_BACKWRAD_CASE(4); break;
        case 5: EXPAND_OP_PREVIOUS_BACKWRAD_CASE(5); break;
        case 6: EXPAND_OP_PREVIOUS_BACKWRAD_CASE(6); break;
        default: break;
      }
    } else {
      for (uint32_t i = 0; i < B; ++i) {
        uint32_t offset = pack_ptr ? pack_ptr[i * 2 + 0] : i * (M / B);
        uint32_t num    = pack_ptr ? pack_ptr[i * 2 + 1] : (M / B);
        auto go_p       = grad_o_ptr + offset * C;
        auto x_p        = x_ptr + offset * C;
        if (first_type == 2 && grad_f_ptr != nullptr) {
          for (uint32_t c = 0; c < C; ++c) {
            if (op == 0) grad_f_ptr[i * C + c] = go_p[c];
            if (op == 1) grad_f_ptr[i * C + c] = go_p[c];
            if (op == 2) grad_f_ptr[i * C + c] = -go_p[c];
            if (op == 3) grad_f_ptr[i * C + c] = x_p[c] * go_p[c];
            if (op == 4) grad_f_ptr[i * C + c] = (first_ptr[i * C + c] < x_p[c] ? go_p[c] : scalar_t(0));
            if (op == 5) grad_f_ptr[i * C + c] = (first_ptr[i * C + c] > x_p[c] ? go_p[c] : scalar_t(0));
            if (op == 6) grad_f_ptr[i * C + c] = scalar_t(0.5) * go_p[c];
          }
        }
        if (grad_x_ptr == nullptr) continue;
        auto gx_p = grad_x_ptr + offset * C;
        if (first_type == 0) {
          for (uint32_t c = 0; c < C; ++c) {
            if (op == 0) gx_p[c] += go_p[c];
            if (op == 1) gx_p[c] += go_p[c];
            if (op == 2) gx_p[c] += -go_p[c];
            if (op == 3) gx_p[c] += x_p[c] * go_p[c];
            if (op == 4) gx_p[c] += (x_p[c] < x_p[c] ? go_p[c] : scalar_t(0));
            if (op == 5) gx_p[c] += (x_p[c] > x_p[c] ? go_p[c] : scalar_t(0));
            if (op == 6) gx_p[c] += scalar_t(0.5) * go_p[c];
          }
        }
        for (uint32_t j = 1; j < num; ++j) {
          for (uint32_t c = 0; c < C; ++c) {
            if (op == 0) gx_p[(j - 1) * C + c] += go_p[j * C + c];
            if (op == 1) gx_p[(j - 1) * C + c] += go_p[j * C + c];
            if (op == 2) gx_p[(j - 1) * C + c] -= go_p[j * C + c];
            if (op == 3) gx_p[(j - 1) * C + c] += go_p[j * C + c] * x_p[j * C + c];
            if (op == 4) gx_p[(j - 1) * C + c] += x_p[(j - 1) * C + c] < x_p[j * C + c] ? go_p[j * C + c] : scalar_t(0);
            if (op == 5) gx_p[(j - 1) * C + c] += x_p[(j - 1) * C + c] > x_p[j * C + c] ? go_p[j * C + c] : scalar_t(0);
            if (op == 6) gx_p[(j - 1) * C + c] += go_p[j * C + c] * scalar_t(0.5);
          }
        }
        for (uint32_t j = 0; j < num; ++j) {
          for (uint32_t c = 0; c < C; ++c) {
            scalar_t last;
            if (j == 0) {
              if (first_type == 0)
                last = x_p[c];
              else if (first_type == 1)
                last = first_value;
              else
                last = first_ptr[i * C + c];
            } else
              last = x_p[(j - 1) * C + c];
            if (op == 0) gx_p[j * C + c] += 0;
            if (op == 1) gx_p[j * C + c] += go_p[j * C + c];
            if (op == 2) gx_p[j * C + c] += go_p[j * C + c];
            if (op == 3) gx_p[j * C + c] += go_p[j * C + c] * last;
            if (op == 4) gx_p[j * C + c] += last < x_p[j * C + c] ? scalar_t(0) : go_p[j * C + c];
            if (op == 5) gx_p[j * C + c] += last > x_p[j * C + c] ? scalar_t(0) : go_p[j * C + c];
            if (op == 6) gx_p[j * C + c] += go_p[j * C + c] * scalar_t(0.5);
          }
        }
      }
    }
  }));
  return {grad_value, grad_first};
}

REGIST_PYTORCH_EXTENSION(nerf_op_previous, {
  m.def("op_previous_forward", &op_previous_forward, "op_previous_forward (CPU & CUDA)");
  m.def("op_previous_backward", &op_previous_backward, "op_previous_backward (CPU & CUDA)");
});