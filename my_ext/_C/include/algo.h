#include <vector>
using std::vector;

struct DisjointSet {
  DisjointSet(int N) { init(N); }
  void init(int N) {
    father.resize(N);
    rank.resize(N);
    for (int i = 0; i < N; ++i) father[i] = i, rank[i] = 0;
  }
  int find(int x) { return father[x] == x ? x : father[x] = find(father[x]); }
  void gather(int x, int y) {
    x = find(x);
    y = find(y);
    if (x == y) return;
    if (rank[x] > rank[y])
      father[y] = x;
    else {
      if (rank[x] == rank[y]) rank[y]++;
      father[x] = y;
    }
  }
  bool same(int x, int y) { return find(x) == find(y); }

  vector<int> father, rank;
};

struct Tree {
  vector<int> head, to, next;
  int num_edges;
  int num_nodes;

  Tree(int M) : head(M, -1), to(M * 2), next(M * 2), num_edges(0) { num_nodes = M; }

  void add_edge(int u, int v) {
    to[num_edges]   = v;
    next[num_edges] = head[u];
    head[u]         = num_edges++;
  }
};