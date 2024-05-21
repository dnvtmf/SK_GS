// Boruvka Superpixel: split image into superpixels
// full rewrite after:  Wei, Yang, Gong, Ahuja, Yang: Superpixel Hierarchy,
//         IEEE Trans Image Proc, 2018
// based on Boruvka's minimum spanning tree algorithm

#include <float.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "common.hpp"

namespace BorukaSuperPixel {
#define CHECK_VERT(v)      \
  {                        \
    assert((v) >= 0);      \
    assert((v) < n_vert_); \
  }

#define CHECK_EDGE(edge)  \
  {                       \
    CHECK_VERT(edge->v0); \
    CHECK_VERT(edge->v1); \
  }

#define PENALTY 1e-8
#define MIN_VALUE 1e-50

struct Edge {
  int v0;                 // vertex 0
  int v1;                 // vertex 1
  struct Edge *v0_next;   // next edge on v0
  struct Edge *v1_next;   // next edge on v1
  float affinity;         // affinity between v0 and v1
  float border_strength;  // at least 1
  float gain_;
};

class BoruvkaSuperpixel {
 public:
  BoruvkaSuperpixel() : dim_(0), avg_data_(nullptr), edgeless_(nullptr) {}
  ~BoruvkaSuperpixel() { clean(); }

  // data access
  int dim() const { return dim_; }
  int *shape()  // treat it as const int *; written non-const for cython
  {
    return shape_;
  }
  int n_vert() const  // number of vertices
  {
    return n_vert_;
  }
  int min_n_supix() const  // minimum number of superpixels
  {
    return n_vert_ - n_mst_;
  }
  int n_edgeless() const  // number of edgeless vertices
  {
    return n_edgeless_;
  }

  template <typename T>
  void build(int num_verites, int num_edges, int n_feature_channels, int32_t *edge, T const *feature,
      T const *affinities, T const *border_strength);

  template <typename T>
  void build_2d(int shape_x, int shape_y, int n_feature_channels, T const *feature, T const *affinities,
      T const *border_strength, int8_t const *edgeless = nullptr, int32_t const *seed = nullptr);
  template <typename T>
  void build_3d(
      int shape_x, int shape_y, int shape_z, int n_feature_channels, T const *feature, T const *border_strength);
  template <typename T>
  void build_3d_of(int shape_x, int shape_y, int shape_z, int n_feature_channels, T const *feature,
      T const *border_strength, int16_t const *optical_flow, double ofedge_prefactor);
  template <typename T>
  void build_3d_of2(int shape_x, int shape_y, int shape_z, int n_feature_channels, T const *feature,
      T const *border_strength, int16_t const *optical_flow, int16_t const *optical_flow_reverse,
      double ofedge_prefactor, int of_tolerance_sq, double of_rel_tolerance = 0.);

  // calculate labels or average
  // return array owned by class, contents changes on subsequent call
  int32_t *label(int n_supix);
  template <typename T>
  void label_o(int n_supix, T *label);
  void seed_id(int32_t *result);
  template <typename T>
  T *average(int n_supix, int n_channels, T const *data);

 private:
  // global data structures
  int dim_;         // dim of pixel array; features are extra
  int shape_[4];    // shape of pixel array
  int stride_[4];   // stride of pixel array
  int n_vert_;      // number of graph vertices (ie. pixels)
  int n_tree_;      // number of trees
  int n_mst_;       // number of mst edges
  int n_edgeless_;  // number of edgeless vertices
  int32_t *label_;  // per vertex: tree label [0 .. n_tree_-1]
  int32_t *seed_;   // per vertex: seed cluster index or negative
  int *parent_;     // per vertex: parent or root vertex within tree
  int *mst_v0_;     // v0 vertex of MST edge, final size=n_vert_-1
  int *mst_v1_;     // v1 vertex of MST edge, final size=n_vert_-1
  void *avg_data_;  // output of average()
  // build-only data structures
  int n_feat_ch_;           // number of feature channels
  int n_edge_;              // number of inter-tree edges in edge_
  int8_t const *edgeless_;  // nullptr or per vertex: bool no edges on vertex
  float *feature_;          // n_feat_ch_ channels per vertex: features
  float *affinities_;       // per vertex: learned affinities
  float *border_strength_;  // per vertex: border strength
  Edge *edge_store_;        // array of all edges
  Edge **edge_;             // inter-tree edges
  Edge **edges_head_;       // per vertex: head of Edge linked list

  // inline
  void add_edge(int v0, int v1, float affinity, float border_prefactor = 1.);
  void add_graph_edge(int v0, int stride, double border_prefactor = 1.);
  void add_graph_edge_conn8(int v0, int v1, int dir_v0, int dir_v1);
  void add_edges();
  void add_edges_2d(int dx0, int dx1, int dy0, int dy1, int stride);
  void add_edges_3d(int dx0, int dx1, int dy0, int dy1, int dz0, int dz1, int stride);
  double calc_dist(Edge *edge, int boruvka_iter);
  double calc_gauss_sim(Edge *edge, int boruvka_iter);
  double ComputeERGain(double wij, double ci, double cj);
  double ComputeBGain(int nVertices, int si, int sj, double Bweight);
  int find_root(int v);
  Edge *find_edge(Edge *edge, int v0, int v1);
  // in cpp
  template <typename T>
  void build_common(
      int n_edge_store, T const *feature, T const *affinities, T const *border_strength, int32_t const *seed = nullptr);
  void build_hierarchy();  // final part of build
  void clean();
};

// ****************************************************************************
inline double BoruvkaSuperpixel::ComputeERGain(double wij, double ci, double cj) {
  double er =
      ((wij + ci) * log(wij + ci) + (wij + cj) * log(wij + cj) - ci * log(ci) - cj * log(cj) - 2 * wij * log(wij)) /
      log(2.0);
  // double er = wij;
  if (er != er) {
    return 0;
  } else {
    return er;
  }
}

inline double BoruvkaSuperpixel::ComputeBGain(int nVertices, int si, int sj, double Bweight) {
  double Si = si * 1.0 / nVertices;
  double Sj = sj * 1.0 / nVertices;

  // double b = (-(Si+Sj)*log(Si+Sj) + Si*log(Si) + Sj*log(Sj))/log(2.0) + 1.0;
  double b = ((-(Si + Sj) * log(Si + Sj) + Si * log(Si) + Sj * log(Sj)) / log(2.0)) * Bweight + 1.0;
  // double b = 1.0;

  return b;
}
// ****************************************************************************
// ****************************************************************************

inline void BoruvkaSuperpixel::add_graph_edge(int v0, int stride, double border_prefactor) {
  int v1 = v0 + stride;

  if (edgeless_ and (edgeless_[v0] or edgeless_[v1])) {
    return;
  }

  Edge *edge     = edge_store_ + n_edge_;  // pointer to next available edge
  edge_[n_edge_] = edge;
  n_edge_++;

  edge->v0 = v0;
  edge->v1 = v1;

  // add the learned affinities (0,1,2,3 correspond to east, south, west, north)
  if (stride == 1)  // horizontal edge
  {
    edge->affinity = (affinities_[8 * edge->v0 + 0] + affinities_[8 * edge->v1 + 2]) / 2.0;
  } else {  // vertical edge
    edge->affinity = (affinities_[8 * edge->v0 + 1] + affinities_[8 * edge->v1 + 3]) / 2.0;
  }

  // insert at head of edge-list for both trees
  edge->v0_next   = edges_head_[v0];
  edge->v1_next   = edges_head_[v1];
  edges_head_[v0] = edge;
  edges_head_[v1] = edge;

  // calc border_strength:
  // min of border_strength at the two vertices, but at least 1
  edge->border_strength = border_prefactor * std::max(std::min(border_strength_[v0], border_strength_[v1]), (float) 1.);

  CHECK_EDGE(edge)
}

inline void BoruvkaSuperpixel::add_edge(int v0, int v1, float affinity, float border_strength) {
  Edge *edge     = edge_store_ + n_edge_;  // pointer to next available edge
  edge_[n_edge_] = edge;
  n_edge_++;

  edge->v0 = v0;
  edge->v1 = v1;

  // insert at head of edge-list for both trees
  edge->v0_next   = edges_head_[v0];
  edge->v1_next   = edges_head_[v1];
  edges_head_[v0] = edge;
  edges_head_[v1] = edge;

  edge->affinity        = affinity;
  edge->border_strength = std::max(border_strength, 1.0f);

  CHECK_EDGE(edge)
}

inline void BoruvkaSuperpixel::add_graph_edge_conn8(int v0, int v1, int dir_v0, int dir_v1)
// v0, v1: node v0 and v1
// dir_v0, dir_v1: the pointing direction of the edge connected to node v0 and v1
{
  if (edgeless_ and (edgeless_[v0] or edgeless_[v1])) {
    return;
  }
  add_edge(v0, v1, (affinities_[8 * v0 + dir_v0] + affinities_[8 * v1 + dir_v1]) / 2.0,
      std::max(border_strength_[v0], border_strength_[v1]));
}

inline void BoruvkaSuperpixel::add_edges() {
  int v0, v1;

  for (int y = 0; y < shape_[0]; y++) {    // height
    for (int x = 0; x < shape_[1]; x++) {  // width

      v0 = y * shape_[1] + x;

      // horizontal edge weight
      if (x < (shape_[1] - 1)) {
        v1 = v0 + 1;
        add_graph_edge_conn8(v0, v1, 0, 4);  // toward right
      }

      // vertical edge weight
      if (y < (shape_[0] - 1)) {
        v1 = v0 + shape_[1];
        add_graph_edge_conn8(v0, v1, 2, 6);  // toward bottom
      }

      // diagonal: toward south-east
      if (x < (shape_[1] - 1) && y < (shape_[0] - 1)) {
        v1 = v0 + shape_[1] + 1;
        add_graph_edge_conn8(v0, v1, 1, 5);
      }

      // diagonal: toward north-east
      if (x < (shape_[1] - 1) && y > 0) {
        v1 = v0 - shape_[1] + 1;
        add_graph_edge_conn8(v0, v1, 7, 3);
      }
    }
  }
}

#define IDX2(x, y) ((x) *stride_[0] + (y) *stride_[1])
#define IDX3(x, y, z) ((x) *stride_[0] + (y) *stride_[1] + (z) *stride_[2])

inline void BoruvkaSuperpixel::add_edges_2d(int dx0, int dx1, int dy0, int dy1, int stride) {
  // nonstandard indent for cleaner look
  for (int x = dx0; x < shape_[0] + dx1; x++) {
    for (int y = dy0; y < shape_[1] + dy1; y++) {
      add_graph_edge(IDX2(x, y), stride);
    }
  }
}

inline void BoruvkaSuperpixel::add_edges_3d(int dx0, int dx1, int dy0, int dy1, int dz0, int dz1, int stride) {
  // nonstandard indent for cleaner look
  for (int x = dx0; x < shape_[0] + dx1; x++) {
    for (int y = dy0; y < shape_[1] + dy1; y++) {
      for (int z = dz0; z < shape_[2] + dz1; z++) {
        add_graph_edge(IDX3(x, y, z), stride);
      }
    }
  }
}

inline double BoruvkaSuperpixel::calc_dist(Edge *edge, int boruvka_iter) {
  double dist = 0;

  if (seed_) {
    int seed0 = seed_[edge->v0];
    int seed1 = seed_[edge->v1];
    if (seed0 >= 0 and seed0 == seed1) {
      // vertices belong to same seed
      // edges with negative weight connect early
      return -(edge->v0 + abs(edge->v1 - edge->v0));
    }
    if (seed0 >= 0 and seed1 >= 0 and seed0 != seed1) {
      // vertices/trees belong to different seed
      // increase weight to postpone connection as much as possible
      dist = PENALTY;
    }
  }

  for (int c = 0; c < n_feat_ch_; c++) {  // all feature channels
    dist += fabsf(feature_[n_feat_ch_ * edge->v0 + c] - feature_[n_feat_ch_ * edge->v1 + c]);
  }

  // unconditionally multiply with border_strength
  return (dist + MIN_VALUE) * (edge->border_strength + MIN_VALUE);
}

inline double BoruvkaSuperpixel::calc_gauss_sim(Edge *edge, int boruvka_iter) {
  double sum       = 0;
  double gauss_sim = 0;    // Gaussian similarity
  double sigma     = 0.5;  // bandwidth parameter in the Gaussian similarity

  if (seed_) {
    int seed0 = seed_[edge->v0];
    int seed1 = seed_[edge->v1];

    if (seed0 >= 0 and seed0 == seed1) {
      // vertices belong to same seed
      // edges with negative weight connect early

      // return -(edge->v0 + abs(edge->v1 - edge->v0));
      gauss_sim = 1;
    }

    if (seed0 >= 0 and seed1 >= 0 and seed0 != seed1) {
      // vertices/trees belong to different seed
      // increase weight to postpone connection as much as possible
      gauss_sim = PENALTY;
    }
  }

  for (int c = 0; c < n_feat_ch_; c++) {  // all feature channels
    sum += pow(feature_[n_feat_ch_ * edge->v0 + c] - feature_[n_feat_ch_ * edge->v1 + c], 2);
  }

  gauss_sim = exp(-sqrt(sum) / (2.0 * pow(sigma, 2)));

  // unconditionally multiply with border_strength
  return gauss_sim;
}

inline int BoruvkaSuperpixel::find_root(int v) {
  // find root by parent_[]
  // use path compression: intermediate nodes' parent are set to ultimate root
  int par = parent_[v];
  if (par != v) {
    parent_[v] = find_root(par);
  }
  return parent_[v];
}

inline Edge *BoruvkaSuperpixel::find_edge(Edge *edge, int v0, int v1) {
  // find an edge connecting tree-roots v0 and v1, following v0's linked list
  while (edge) {
    // edge->v0 == v0 or edge->v1 == v0
    if (edge->v0 == v0) {
      if (edge->v1 == v1) {
        return edge;
      }
      edge = edge->v0_next;
    } else {
      assert(edge->v1 == v0);
      if (edge->v0 == v1) {
        return edge;
      }
      edge = edge->v1_next;
    }
  }
  return nullptr;
}

template <typename T>
void BoruvkaSuperpixel::build(int num_verites, int num_edges, int n_feature_channels, int32_t *edge, T const *feature,
    T const *affinities, T const *border_strength) {
  clean();
  dim_       = 1;
  n_vert_    = num_verites;
  n_feat_ch_ = n_feature_channels;
  edgeless_  = nullptr;
  // init build-only data structures
  n_edge_     = 0;
  edge_store_ = new Edge[num_edges];    // will be filled soon
  edge_       = new Edge *[num_edges];  // likewise
  edges_head_ = new Edge *[num_verites];

  affinities_      = nullptr;
  border_strength_ = nullptr;
  seed_            = nullptr;
  edgeless_        = nullptr;

  feature_ = new float[n_vert_ * n_feat_ch_];          // float array
  for (int vc = 0; vc < n_vert_ * n_feat_ch_; vc++) {  // all vertices&features
    feature_[vc] = feature[vc];                        // conversion
  }
  for (int v = 0; v < n_vert_; v++) {  // all vertices
    edges_head_[v] = nullptr;          // empty linked list
  }

  for (int e = 0; e < num_edges; ++e) {
    add_edge(edge[e * 2 + 0], edge[e * 2 + 1], affinities[e], border_strength[e]);
  }
  assert(n_edge_ == num_edges);
  build_hierarchy();
}

template <typename T>
void BoruvkaSuperpixel::build_2d(int shape_x, int shape_y, int n_feature_channels, T const *feature,
    T const *affinities, T const *border_strength, int8_t const *edgeless, int32_t const *seed) {
  // TODO argument error check
  clean();

  // init global data structures
  dim_       = 2;
  shape_[0]  = shape_x;
  shape_[1]  = shape_y;
  stride_[0] = shape_[1];
  stride_[1] = 1;
  n_vert_    = shape_x * shape_y;

  // init build-only data structures
  n_feat_ch_ = n_feature_channels;
  edgeless_  = edgeless;

  // 8-connected scenario
  int n_edge_store = 4 * (shape_[0] - 1) * (shape_[1] - 1) + shape_[0] + shape_[1] - 2;

  build_common(n_edge_store, feature, affinities, border_strength, seed);

  // build image graph
  add_edges();

  assert(n_edge_ == n_edge_store);

  build_hierarchy();
}

template <typename T>
void BoruvkaSuperpixel::build_3d(
    int shape_x, int shape_y, int shape_z, int n_feature_channels, T const *feature, T const *border_strength) {
  // TODO argument error check
  clean();

  // init global data structures
  dim_       = 3;
  shape_[0]  = shape_x;
  shape_[1]  = shape_y;
  shape_[2]  = shape_z;
  stride_[0] = shape_[1] * shape_[2];
  stride_[1] = shape_[2];
  stride_[2] = 1;
  n_vert_    = shape_x * shape_y * shape_z;

  // init build-only data structures
  n_feat_ch_ = n_feature_channels;
  /*
  int n_edge_store =  // number of graph edges
      (shape_[0] - 1) * shape_[1] * shape_[2] +
      (shape_[1] - 1) * shape_[2] * shape_[0] +
      (shape_[2] - 1) * shape_[0] * shape_[1];*/
  // build_common(n_edge_store, feature, affinities, border_strength);

  // build image graph: add edges
  add_edges_3d(0, -1, 0, 0, 0, 0, stride_[0]);  // x+
  add_edges_3d(0, 0, 0, -1, 0, 0, stride_[1]);  // y+
  add_edges_3d(0, 0, 0, 0, 0, -1, stride_[2]);  // z+

  //    assert(n_edge_ == n_edge_store);

  build_hierarchy();
}

template <typename T>
void BoruvkaSuperpixel::build_3d_of(int shape_x, int shape_y, int shape_z, int n_feature_channels, T const *feature,
    T const *border_strength, int16_t const *optical_flow, double ofedge_prefactor) {
  // TODO argument error check
  clean();

  // init global data structures
  dim_       = 3;
  shape_[0]  = shape_x;
  shape_[1]  = shape_y;
  shape_[2]  = shape_z;
  stride_[0] = shape_[1] * shape_[2];
  stride_[1] = shape_[2];
  stride_[2] = 1;
  n_vert_    = shape_x * shape_y * shape_z;

  // init build-only data structures
  n_feat_ch_ = n_feature_channels;
  /*
  int n_edge_store =  // number of graph edges
      (shape_[0] - 1) * shape_[1] * shape_[2] +
      (shape_[1] - 1) * shape_[2] * shape_[0] +
      // and at most 1 edge in (z+ + o.f.) direction
      (shape_[2] - 1) * shape_[0] * shape_[1];*/
  // build_common(n_edge_store, feature, affinities, border_strength);

  // build image graph: add edges
  add_edges_3d(0, -1, 0, 0, 0, 0, stride_[0]);  // x+
  add_edges_3d(0, 0, 0, -1, 0, 0, stride_[1]);  // y+
  // add_edges_3d( 0,  0,  0,  0,  0, -1, stride_[2]);   // z+
  //  add optical flow edges
  for (int x = 0; x < shape_[0]; x++) {
    for (int y = 0; y < shape_[1]; y++) {
      for (int z = 0; z < shape_[2] - 1; z++) {
        int dx = optical_flow[2 * IDX3(x, y, z)];
        int dy = optical_flow[2 * IDX3(x, y, z) + 1];
        int x1 = x + dx;
        int y1 = y + dy;
        if (0 <= x1 and x1 < shape_[0] and 0 <= y1 and y1 < shape_[1]) {
          int stride = IDX3(dx, dy, 1);
          add_graph_edge(IDX3(x, y, z), stride, ofedge_prefactor);
        }
      }
    }
  }
  // assert(n_edge_ <= n_edge_store);    // less if o.f. points outside

  build_hierarchy();
}

template <typename T>
void BoruvkaSuperpixel::build_3d_of2(int shape_x, int shape_y, int shape_z, int n_feature_channels, T const *feature,
    T const *border_strength, int16_t const *optical_flow, int16_t const *optical_flow_reverse, double ofedge_prefactor,
    int of_tolerance_sq, double of_rel_tolerance) {
  // TODO argument error check
  clean();

  // init global data structures
  dim_       = 3;
  shape_[0]  = shape_x;
  shape_[1]  = shape_y;
  shape_[2]  = shape_z;
  stride_[0] = shape_[1] * shape_[2];
  stride_[1] = shape_[2];
  stride_[2] = 1;
  n_vert_    = shape_x * shape_y * shape_z;

  // init build-only data structures
  n_feat_ch_ = n_feature_channels;
  /*
  int n_edge_store =  // number of graph edges
      (shape_[0] - 1) * shape_[1] * shape_[2] +
      (shape_[1] - 1) * shape_[2] * shape_[0] +
      // and at most 1 edge in (z+ + o.f.) direction
      (shape_[2] - 1) * shape_[0] * shape_[1];*/
  // build_common(n_edge_store, feature, affinities, border_strength);

  // build image graph: add edges
  add_edges_3d(0, -1, 0, 0, 0, 0, stride_[0]);  // x+
  add_edges_3d(0, 0, 0, -1, 0, 0, stride_[1]);  // y+
  // add_edges_3d( 0,  0,  0,  0,  0, -1, stride_[2]);   // z+
  //  add optical flow edges
  for (int x = 0; x < shape_[0]; x++) {
    for (int y = 0; y < shape_[1]; y++) {
      for (int z = 0; z < shape_[2] - 1; z++) {
        int dx = optical_flow[2 * IDX3(x, y, z)];
        int dy = optical_flow[2 * IDX3(x, y, z) + 1];
        int x1 = x + dx;
        int y1 = y + dy;
        if (0 <= x1 and x1 < shape_[0] and 0 <= y1 and y1 < shape_[1]) {
          int dx_r         = optical_flow_reverse[2 * IDX3(x1, y1, z)];
          int dy_r         = optical_flow_reverse[2 * IDX3(x1, y1, z) + 1];
          int d_sq         = dx * dx + dy * dy;
          int d_r_sq       = dx_r * dx_r + dy_r * dy_r;
          int tolerance_sq = std::max({of_tolerance_sq, (int) (of_rel_tolerance * of_rel_tolerance * d_sq),
              (int) (of_rel_tolerance * of_rel_tolerance * d_r_sq)});
          int errx         = dx + dx_r;
          int erry         = dy + dy_r;
          if (errx * errx + erry * erry <= tolerance_sq) {
            int stride = IDX3(dx, dy, 1);
            add_graph_edge(IDX3(x, y, z), stride, ofedge_prefactor);
          }
        }
      }
    }
  }
  // assert(n_edge_ <= n_edge_store);    // less if o.f. points outside

  build_hierarchy();
}

template <typename T>
void BoruvkaSuperpixel::label_o(int n_supix, T *label) {
  // TODO error check on n_supix

  int beg = n_vert_ - n_tree_;
  int end = n_vert_ - n_supix;
  if (end < beg) {
    // more superpixels: start fresh, every pixel is superpixel
    for (int v = 0; v < n_vert_; v++) {  // for all vertices
      parent_[v] = v;
    }
    beg = 0;
  }

  // decrease number of superpixels:
  // further connect trees via parent_ according to subsequent edges
  for (int e = beg; e < end; e++) {  // loop on mst edges
    int root0 = find_root(mst_v0_[e]);
    int root1 = find_root(mst_v1_[e]);
    // ensure root_of_vertex <= vertex
    if (root0 < root1) {
      parent_[root1] = root0;
    } else {
      parent_[root0] = root1;
    }
  }

  // supix label: number sequentially the trees
  n_tree_ = 0;
  for (int v = 0; v < n_vert_; v++) {  // for all vertices
    int root = find_root(v);
    if (v == root) {         // first encounter with this tree
      label[v] = n_tree_++;  // take next label
    } else {
      label[v] = label[root];  // label was taken at root vertex
    }
  }
  assert(n_tree_ == n_supix);
}

int32_t *BoruvkaSuperpixel::label(int n_supix) {
  if (n_supix == 0) {
    // only makes sense when n_mst_ < n_vert_-1,
    // eg. when seeded, and penalized merges are not executed
    n_supix = min_n_supix();
  }
  if (n_supix < min_n_supix()) {
    return nullptr;
  }
  // too large n_supix silently returns the original image

  int beg = n_vert_ - n_tree_;
  int end = n_vert_ - n_supix;
  if (end < beg) {
    // more superpixels: start fresh, every pixel is superpixel
    for (int v = 0; v < n_vert_; v++) {  // for all vertices
      parent_[v] = v;
    }
    beg = 0;
  }

  // decrease number of superpixels:
  // further connect trees via parent_ according to subsequent edges
  for (int e = beg; e < end; e++) {  // loop on mst edges
    int root0 = find_root(mst_v0_[e]);
    int root1 = find_root(mst_v1_[e]);
    // ensure root_of_vertex <= vertex
    if (root0 < root1) {
      parent_[root1] = root0;
    } else {
      parent_[root0] = root1;
    }
  }

  // supix label: number sequentially the trees
  n_tree_ = 0;
  for (int v = 0; v < n_vert_; v++) {  // for all vertices
    int root = find_root(v);
    if (v == root) {          // first encounter with this tree
      label_[v] = n_tree_++;  // take next label
    } else {
      label_[v] = label_[root];  // label was taken at root vertex
    }
  }
  assert(n_tree_ == n_supix);
  return label_;  // array owned by *this
}

void BoruvkaSuperpixel::seed_id(int32_t *result) {
  if (seed_) {
    for (int v = 0; v < n_vert_; v++) {  // for all vertices
      result[v] = seed_[find_root(v)];
    }
  } else {
    // invalidate seed_id for all pixels
    for (int v = 0; v < n_vert_; v++) {  // for all vertices
      result[v] = -1;
    }
  }
}

template <typename T>
T *BoruvkaSuperpixel::average(int n_supix, int n_channels, T const *data) {
  if (n_supix == 0) {
    // only makes sense when n_mst_ < n_vert_-1,
    // eg. when seeded, and penalized merges are not executed
    n_supix = min_n_supix();
  }

  // prepare output array
  if (avg_data_) {
    free(avg_data_);
  }
  avg_data_ = malloc(n_channels * n_vert_ * sizeof(T));

  int32_t *ret = label(n_supix);
  if (not ret) {
    return nullptr;
  }

  // perform average
  int *count = new int[n_supix];
  float *sum = new float[n_supix * n_channels];
  memset(count, 0, sizeof(int) * n_supix);
  memset(sum, 0, sizeof(float) * n_supix * n_channels);
  for (int v = 0; v < n_vert_; v++) {
    count[label_[v]]++;
    for (int c = 0; c < n_channels; c++) {
      sum[n_channels * label_[v] + c] += data[n_channels * v + c];
    }
  }
  // write to output
  for (int v = 0; v < n_vert_; v++) {
    for (int c = 0; c < n_channels; c++) {
      ((T *) avg_data_)[n_channels * v + c] = sum[n_channels * label_[v] + c] / count[label_[v]];
    }
  }

  delete[] sum;
  delete[] count;
  return (T *) avg_data_;
}

#define INSTANTIATE(T)                                                                                                \
  template void BoruvkaSuperpixel::build<T>(int, int, int, int32_t *, T const *, T const *, T const *);               \
  template void BoruvkaSuperpixel::build_2d<T>(                                                                       \
      int, int, int, T const *, T const *, T const *, int8_t const *, int32_t const *);                               \
  template void BoruvkaSuperpixel::build_3d<T>(int, int, int, int, T const *, T const *);                             \
  template void BoruvkaSuperpixel::build_3d_of<T>(int, int, int, int, T const *, T const *, int16_t const *, double); \
  template void BoruvkaSuperpixel::build_3d_of2<T>(                                                                   \
      int, int, int, int, T const *, T const *, int16_t const *, int16_t const *, double, int, double);               \
  template T *BoruvkaSuperpixel::average<T>(int, int, T const *);                                                     \
  template void BoruvkaSuperpixel::label_o<T>(int, T *);

INSTANTIATE(uint8_t)
INSTANTIATE(uint16_t)
INSTANTIATE(int8_t)
INSTANTIATE(int16_t)
INSTANTIATE(int32_t)
INSTANTIATE(float)
INSTANTIATE(double)

// ******************************* internals **********************************

void BoruvkaSuperpixel::clean() {
  if (dim_) {
    // allocated by average
    if (avg_data_) {
      free(avg_data_);
      avg_data_ = nullptr;
    }
    // allocated by build_hierarchy
    delete[] mst_v1_;
    delete[] mst_v0_;
    delete[] parent_;
    delete[] label_;
    // allocated by build_common
    if (seed_) {
      delete[] seed_;
      seed_ = nullptr;
    }

    dim_ = 0;
  }
}

template <typename T>
void BoruvkaSuperpixel::build_common(
    int n_edge_store, T const *feature, T const *affinities, T const *border_strength, int32_t const *seed) {
  // init build-only data structures
  n_edge_     = 0;
  edge_store_ = new Edge[n_edge_store];    // will be filled soon
  edge_       = new Edge *[n_edge_store];  // likewise
  edges_head_ = new Edge *[n_vert_];

  for (int v = 0; v < n_vert_; v++) {  // all vertices
    edges_head_[v] = nullptr;          // empty linked list
  }

  // convert input arrays to float
  feature_         = new float[n_vert_ * n_feat_ch_];  // float array
  border_strength_ = new float[n_vert_];               // float array
  affinities_      = new float[n_vert_ * 8];           // float array

  for (int vc = 0; vc < n_vert_ * n_feat_ch_; vc++) {  // all vertices&features
    feature_[vc] = feature[vc];                        // conversion
  }
  for (int vv = 0; vv < n_vert_ * 8; vv++) {  // all vertices&affinities
    affinities_[vv] = affinities[vv];         // conversion
  }
  for (int v = 0; v < n_vert_; v++) {          // all vertices
    border_strength_[v] = border_strength[v];  // conversion
  }

  // calc n_edgeless
  n_edgeless_ = 0;
  if (edgeless_) {
    for (int v = 0; v < n_vert_; v++) {
      if (edgeless_[v]) {
        n_edgeless_++;
      }
    }
  }

  // seed: alloc only if supplied
  if (seed) {
    seed_ = new int32_t[n_vert_];
    for (int v = 0; v < n_vert_; v++) {  // all vertices
      seed_[v] = seed[v];
    }
  } else {
    seed_ = nullptr;
  }
}

void BoruvkaSuperpixel::build_hierarchy() {
  // init global data structures
  n_tree_ = n_vert_;  // initially each vertex is a tree
  label_  = new int[n_vert_];
  parent_ = new int[n_vert_];
  mst_v0_ = new int[n_vert_];  // no need to initialize
  mst_v1_ = new int[n_vert_];

  // all vertices
  for (int v = 0; v < n_vert_; v++) {
    label_[v]  = v;
    parent_[v] = v;
  }

  // init build-only data structures
  n_mst_         = 0;
  int *tree_root = new int[n_vert_];  // per tree label: root vertex
  int *tree_size = new int[n_vert_];  // per root vertex: tree size

  //------------------------------------------------------------------------//
  // initialise loop weight
  double *loop = new double[n_vert_];
  for (int i = 0; i < n_vert_; i++) {
    loop[i] = 0;
  }

  for (int i = 0; i < n_edge_; i++) {
    Edge *edge = edge_[i];
    loop[edge->v0] += edge->affinity;
    loop[edge->v1] += edge->affinity;
  }

  // compute total weight
  double wT = 0;
  for (int i = 0; i < n_vert_; i++) {
    wT += loop[i];
  }

  // normalise the weights
  for (int i = 0; i < n_edge_; i++) {
    Edge *edge = edge_[i];
    (edge->affinity) /= wT;
    edge_[i] = edge;
  }

  for (int i = 0; i < n_vert_; i++) {
    loop[i] /= wT;
  }

  // Compute initial gain and decide the weighting on the balancing term
  double *erGainArr = new double[n_edge_];  // gain in entropy rate term

  for (int i = 0; i < n_edge_; i++) {
    Edge *edge = edge_[i];

    erGainArr[i] = ComputeERGain(edge->affinity, loop[edge->v0] - edge->affinity, loop[edge->v1] - edge->affinity);

    edge->gain_ = erGainArr[i];
    edge_[i]    = edge;
  }

  delete[] erGainArr;

  //------------------------------------------------------------------------//

  typedef struct {
    double value;  // value of best outgoing edge
    Edge *edge;    // best outgoing edge
  } MinPair_t;

  MinPair_t *min_pair = new MinPair_t[n_vert_];  // will be used for sorting

  for (int v = 0; v < n_vert_; v++) {  // all vertices
    edges_head_[v] = nullptr;          // empty linked list
    tree_root[v]   = v;
    tree_size[v]   = 1;
  }

  // initialise the edge weights between nodes using the learned affinities

  // build Boruvka minimum spanning tree hierarchy
  for (int boruvka_iter = 0; n_tree_ > 1; boruvka_iter++) {
    // STEP 1
    // find minimal outgoing edge for each tree
    for (int t = 0; t < n_tree_; t++) {  // all trees
      min_pair[t].value = MIN_VALUE;     // invalidate distance
    }

    for (int e = 0; e < n_edge_; e++) {  // all inter-tree edges
      Edge *edge = edge_[e];

      int label0 = label_[edge->v0];  // label: 0 .. n_tree_-1
      int label1 = label_[edge->v1];

      double dist_feat = calc_dist(edge, boruvka_iter);
      double dist      = edge->gain_ / dist_feat;  // entropy rate and RGB info
      // double dist = edge->gain_; // entropy rate only

      if (dist > min_pair[label0].value) {
        min_pair[label0].value = dist;
        min_pair[label0].edge  = edge;
      }

      if (dist > min_pair[label1].value) {
        min_pair[label1].value = dist;
        min_pair[label1].edge  = edge;
      }
    }

    // STEP 2
    // connect trees with min outgoing edges
    // update: mst_v0_, mst_v1_, n_mst_, parent_, seed_
    std::sort(
        min_pair, min_pair + n_tree_, [](const MinPair_t &a, const MinPair_t &b) -> bool { return a.value > b.value; });

    // within a Boruvka iteration: connect weak edges earlier
    bool all_penalized = (min_pair[0].value <= PENALTY);

    if (all_penalized) {
      // break out of boruvka_iter loop
      break;
    }

    double erGain;
    for (int t = 0; t < n_tree_; t++) {  // all min outgoing edges

      if (min_pair[t].value <= MIN_VALUE) {
        break;
      }

      Edge *edge = min_pair[t].edge;
      int root0  = find_root(edge->v0);
      int root1  = find_root(edge->v1);

      if (root0 == root1) {  // reverse link was already added
        continue;
      }

      if ((not all_penalized) and min_pair[t].value <= PENALTY) {
        // leave condition in case later wanted to change code
        // to do penalized merges
        // process penalized edges only when no other left
        continue;
      }

      if (seed_) {
        int seed0 = seed_[root0];
        int seed1 = seed_[root1];
        if (seed0 >= 0 and seed1 >= 0 and seed0 != seed1) {
          continue;
        }
        if (seed0 >= 0 and seed1 < 0) {
          seed_[root1] = seed0;
        }
        if (seed1 >= 0 and seed0 < 0) {
          seed_[root0] = seed1;
        }
      }

      // add edge to MST, connecting the root vertices of the trees
      mst_v0_[n_mst_] = root0;
      mst_v1_[n_mst_] = root1;
      n_mst_++;

      if (root0 < root1) {  // ensure root(v) <= v
        parent_[root1] = root0;
      } else {
        parent_[root0] = root1;
      }

      if (label_[edge->v0] != label_[edge->v1]) {
        // update the loop weights
        loop[edge->v0] -= edge->affinity;
        loop[edge->v1] -= edge->affinity;

        // update the gain of the best edge of each tree
        erGain = ComputeERGain(edge->affinity, loop[edge->v0] - edge->affinity, loop[edge->v1] - edge->affinity);

        edge->gain_ = erGain;

        min_pair[t].edge = edge;
      }
    }

    // STEP 3
    // update: n_tree_, label_, tree_root, edges_head_, feature_
    // (and tree_size, only used here)
    int n_tree_old = n_tree_;  // number of trees in previous iteration
    n_tree_        = 0;        // counting new trees

    for (int t = 0; t < n_tree_old; t++) {  // all old trees

      // std::cout << t << endl;

      int v    = tree_root[t];  // root vertex of old tree
      int root = find_root(v);  // new root

      if (v == root) {
        // first encounter with this new tree
        label_[v]            = n_tree_;
        tree_root[n_tree_]   = v;
        edges_head_[n_tree_] = nullptr;
        n_tree_++;
      } else {
        // this old tree is now appended to another tree
        label_[v]       = label_[root];
        int size        = tree_size[v];
        int size_root   = tree_size[root];
        int size_tot    = size + size_root;
        tree_size[root] = size_tot;

        for (int c = 0; c < n_feat_ch_; c++) {
          // its features are blended into new tree, saved at new root
          feature_[n_feat_ch_ * root + c] =
              (feature_[n_feat_ch_ * v + c] * size + feature_[n_feat_ch_ * root + c] * size_root) / size_tot;
        }
      }
    }
    assert(n_tree_ + n_mst_ == n_vert_);

    // STEP 4
    // update edges: remove intra-tree edges
    // update: n_edge_, edge_, edges_head_

    int n_edge_old = n_edge_;  // number of edges in previous iteration
    n_edge_        = 0;        // counting active edges

    for (int e = 0; e < n_edge_old; e++) {  // all old inter-tree edges

      Edge *edge = edge_[e];
      CHECK_EDGE(edge);

      int v0    = edge->v0;
      int v1    = edge->v1;
      int root0 = parent_[v0];  // all paths are compressed now
      int root1 = parent_[v1];

      assert(root0 == parent_[root0]);
      assert(root1 == parent_[root1]);

      if (root0 == root1) {
        // intra-tree edge: do not keep
        continue;
      }

      if (root1 < root0) {
        std::swap(v0, v1);
        std::swap(root0, root1);
      }  // now root0 < root1

      int label0 = label_[v0];
      int label1 = label_[v1];

      Edge *edge0 = edges_head_[label0];
      Edge *edge1 = edges_head_[label1];

      if (edge0) {
        CHECK_EDGE(edge0);
      }

      if (label0 != label1) {
        // rewire link to connect roots
        // append to edge_ and linked lists
        edge->v0 = root0;
        edge->v1 = root1;
        assert(edge->v0 >= 0);
        assert(edge->v1 >= 0);
        edge->v0_next    = edge0;
        edge->v1_next    = edge1;
        edge_[n_edge_++] = edge;
        CHECK_EDGE(edge)
        edges_head_[label0] = edge;
        edges_head_[label1] = edge;
      }
    }
  }

  // allocated by build_hierarchy
  delete[] tree_root;
  delete[] tree_size;
  delete[] min_pair;
  delete[] loop;

  // allocated by build_common
  if (border_strength_) delete[] border_strength_;
  if (feature_) delete[] feature_;
  if (affinities_) delete[] affinities_;
  delete[] edges_head_;
  delete[] edge_;
  delete[] edge_store_;
}

Tensor get_super_pixel_for_image(Tensor &features, Tensor &affinities, Tensor &border_strength, int num_sp) {
  CHECK_NDIM(features, 3);
  CHECK_NDIM(affinities, 3);

  int H = features.size(0);
  int W = features.size(1);
  int C = features.size(2);
  CHECK_SHAPE(border_strength, H, W);
  CHECK_SHAPE(affinities, H, W, 8);
  BCNN_ASSERT(!features.is_cuda() && !affinities.is_cuda() && !border_strength.is_cuda(), "all Tensor must be cpu");
  BCNN_ASSERT(features.dtype() == affinities.dtype() && features.dtype() == border_strength.dtype(),
      "All tensor must have same dtype.");
  BoruvkaSuperpixel SP;
  Tensor labels = torch::zeros({H, W}, torch::kInt32);
  AT_DISPATCH_ALL_TYPES(features.scalar_type(), "get_super_pixel_for_image", [&] {
    auto feat_ptr  = features.contiguous().data_ptr<scalar_t>();
    auto affi_ptr  = affinities.contiguous().data_ptr<scalar_t>();
    auto bost_ptr  = border_strength.contiguous().data_ptr<scalar_t>();
    int *label_ptr = labels.data_ptr<int>();
    SP.build_2d<scalar_t>(H, W, C, feat_ptr, affi_ptr, bost_ptr, nullptr, nullptr);
    SP.label_o(num_sp, label_ptr);
  });

  return labels;
}

Tensor get_super_pixel_for_graph(
    Tensor &features, Tensor &edges, Tensor &affinities, Tensor &border_strength, int num_sp) {
  CHECK_NDIM(features, 2);

  int N = features.size(0);
  int C = features.size(1);
  int E = edges.size(0);
  CHECK_SHAPE(edges, E, 2);
  CHECK_SHAPE(affinities, E);
  CHECK_SHAPE(border_strength, E);
  BCNN_ASSERT(!features.is_cuda() && !edges.is_cuda() && !affinities.is_cuda() && !border_strength.is_cuda(),
      "All tensors must be cpu");
  BCNN_ASSERT(features.dtype() == affinities.dtype() && features.dtype() == border_strength.dtype(),
      "All tensors must have same dtype.");
  CHECK_TYPE(edges, torch::kInt32);

  BoruvkaSuperpixel SP;
  Tensor labels = torch::zeros({N}, torch::kInt32);
  AT_DISPATCH_ALL_TYPES(features.scalar_type(), "get_super_pixel_for_image", [&] {
    SP.build<scalar_t>(N, E, C, edges.contiguous().data_ptr<int>(), features.contiguous().data_ptr<scalar_t>(),
        affinities.contiguous().data_ptr<scalar_t>(), border_strength.contiguous().data_ptr<scalar_t>());
    SP.label_o(num_sp, labels.data_ptr<int>());
  });

  return labels;
}

REGIST_PYTORCH_EXTENSION(other_super_pixel, {
  m.def("get_super_pixel_for_image", &get_super_pixel_for_image, "get_super_pixel_for_image (CPU)");
  m.def("get_super_pixel_for_graph", &get_super_pixel_for_graph, "get_super_pixel_for_graph (CPU)");
});

}  // namespace BorukaSuperPixel