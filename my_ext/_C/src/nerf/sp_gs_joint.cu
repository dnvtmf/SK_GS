#include "util.cuh"
#include "algo.h"
#include "ops_3d.h"
#include "ops_3d_types.h"
#include <queue>
using namespace OPS_3D;

namespace SP_GS {

void dfs_joint_depth(int u, int fa, int* depth, int* parents, Tree& t, int d, int N) {
  depth[u]           = d;
  parents[u * N + 0] = fa;
  for (int i = 1; i < N; ++i) parents[u * N + i] = parents[parents[u * N + i - 1] * N + i - 1];
  for (int i = t.head[u]; i != -1; i = t.next[i]) {
    int v = t.to[i];
    if (v != fa) {
      dfs_joint_depth(v, u, depth, parents, t, d + 1, N);
    }
  }
}

int find_joint_root(Tree& t, int& max_depth) {
  std::queue<int> que;
  vector<int> visited(t.num_nodes, 0);
  vector<int> num_edges(t.num_nodes, 0);
  for (int i = 0; i < t.num_nodes; ++i) {
    for (int e = t.head[i]; e != -1; e = t.next[e]) {
      num_edges[i]++;
    }
    if (num_edges[i] <= 1) {
      que.push(i);
      visited[i] = 1;
    }
  }
  int root = -1;
  while (!que.empty()) {
    int u = que.front();
    root  = u;
    que.pop();
    for (int i = t.head[u]; i != -1; i = t.next[i]) {
      int v = t.to[i];
      if (num_edges[v] > 1) {
        visited[v] = max(visited[v], visited[u] + 1);
        num_edges[v]--;
        if (num_edges[v] <= 1) que.push(v);
      }
    }
  }

  max_depth = 0;
  for (size_t i = 0; i < visited.size(); ++i) max_depth = max(max_depth, visited[i]);
  return root;
}

tuple<Tensor, Tensor, int> joint_discovery(Tensor dist) {
  int M = dist.size(0);
  CHECK_SHAPE(dist, M, M);
  Tensor order = dist.flatten().argsort().cpu();
  DisjointSet ds(M);
  Tree tree(M);

  int64_t* order_data = order.contiguous().data<int64_t>();
  for (int i = 0, k = 0; i < M - 1; ++i) {
    while (true) {
      int a = order_data[k++];
      int b = a / M;
      a     = a % M;
      if (!ds.same(a, b)) {
        ds.gather(a, b);
        tree.add_edge(a, b);
        tree.add_edge(b, a);
        break;
      }
    }
  }
  int max_depth = 0;
  int max_level = 0;
  int root      = find_joint_root(tree, max_depth);
  while ((1 << max_level) < max_depth) max_level += 1;

  Tensor parents = torch::full({M, max_level}, root, order.options().dtype(torch::kInt32));
  Tensor depth   = torch::zeros(M, parents.options());
  dfs_joint_depth(root, root, depth.data<int32_t>(), parents.data<int32_t>(), tree, 0, max_level);
  return std::make_tuple(parents.to(dist.device()), depth.to(dist.device()), root);
}

#define CHECK_SAME_DEVICE(tensor, ...)                             \
  {                                                                \
    auto device  = tensor.device();                                \
    bool is_same = true;                                           \
    for (auto& t : {__VA_ARGS__}) is_same &= t.device() == device; \
    BCNN_ASSERT(is_same, "all tensor must be same device");        \
  }

template <typename T>
__global__ void skeleton_warp_forward_kernel(int M, int D, T* __restrict__ pos, T* __restrict__ R,
    int32_t* __restrict__ father, int32_t* __restrict__ depth, T* out) {
  int m = threadIdx.x;
  if (m >= M) return;
}

Tensor skeleton_warp(Tensor& joint_t, Tensor& joint_R, Tensor& T, Tensor& connection, Tensor& depth, int max_depth) {
  CHECK_SAME_DEVICE(joint_t, joint_R, T, connection, depth);
  int M = joint_t.size(0);
  CHECK_SHAPE(joint_t, M, 3);
  CHECK_SHAPE(joint_R, M, 3, 3);
  BCNN_ASSERT(joint_R.dtype() == joint_t.dtype(), "dtype for joint_t and joint_R must be same");
  CHECK_SHAPE(connection, M);
  CHECK_SHAPE(depth, M);
  CHECK_TYPE(connection, torch::kInt32);
  CHECK_TYPE(depth, torch::kInt32);

  Tensor output = torch::zeros({M, 4, 4}, joint_R.options());
  BCNN_ASSERT(M <= 1024, "only support M <= 1024 now");
  if (joint_t.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(joint_t.scalar_type(), "skeleton_warp_cuda", [&] {
      skeleton_warp_forward_kernel<scalar_t> KERNEL_ARG(get_cuda_threads(M), 1)(M, max_depth,
          joint_t.contiguous().data_ptr<scalar_t>(), joint_R.contiguous().data_ptr<scalar_t>(),
          connection.contiguous().data_ptr<int32_t>(), depth.contiguous().data_ptr<int32_t>(),
          output.data_ptr<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(joint_t.scalar_type(), "skeleton_warp_cpu", [&] {
      vec3<scalar_t>* pos = (vec3<scalar_t>*) joint_t.contiguous().data_ptr<scalar_t>();
      mat3<scalar_t>* R   = (mat3<scalar_t>*) joint_R.contiguous().data_ptr<scalar_t>();
      int32_t* fa         = connection.contiguous().data_ptr<int32_t>();
      int32_t* d          = depth.contiguous().data_ptr<int32_t>();
      mat4<scalar_t>* out = (mat4<scalar_t>*) output.data_ptr<scalar_t>();
      for (int i = 0; i < M; ++i) {
        mat4<scalar_t> res = mat4<scalar_t>::I();
        int a = i, b = fa[i];
        for (int j = 0; j < min(max_depth, d[i]) && b >= 0; ++j) {
          auto Ri = R[a];
          auto ti = pos[a];
          ti      = ti - (Ri * ti);
          res     = res * mat4<scalar_t>(Ri, ti);

          a = b;
          b = fa[a];
        }
      }
    });
  }
  return output;
}

REGIST_PYTORCH_EXTENSION(sp_gs_joint, { m.def("joint_discovery", &joint_discovery, "joint_discovery (CPU<-CUDA)"); })
}  // namespace SP_GS