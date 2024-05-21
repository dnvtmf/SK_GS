#pragma once
namespace OPS_3D {
#include "cuda.h"
#define FUNC_DECL __device__ __host__ __forceinline__

template <typename T>
struct vec3 {
  union {
    struct {
      T x, y, z;
    };
    T data[3];
  };

  FUNC_DECL vec3() { x = 0, y = 0, z = 0; }
  FUNC_DECL vec3(T x_, T y_, T z_) { x = x_, y = y_, z = z_; }
  FUNC_DECL vec3(const vec3 &b) { x = b.x, y = b.y, z = b.z; }
  FUNC_DECL vec3(T *ptr) { x = ptr[0], y = ptr[1], z = ptr[2]; }
  FUNC_DECL vec3<T> operator*(const T s) { return {s * x, s * y, s * z}; }
  FUNC_DECL T &operator[](const int i) { return data[i]; }
  FUNC_DECL vec3<T> operator+(const vec3<T> &o) { return {x + o.x, y + o.y, z + o.z}; }
  FUNC_DECL vec3<T> operator-(const vec3<T> &o) { return {x - o.x, y - o.y, z - o.z}; }
};

template <typename T>
struct vec4 {
  union {
    struct {
      T x, y, z, w;
    };
    T data[4];
  };
  FUNC_DECL vec4(vec3<T> &a) { x = a.x, y = a.y, z = a.z, w = 0; }
  FUNC_DECL vec4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
  FUNC_DECL T &operator[](const int i) { return data[i]; }
};

template <typename T>
struct mat3 {
  union {
    struct {
      vec3<T> value[3];
    };
    T _data[9];
  };
  FUNC_DECL mat3() {
    for (int i = 0; i < 9; ++i) _data[i] = 0;
  }
  FUNC_DECL mat3(const mat3<T> &b) {
    for (int i = 0; i < 9; ++i) _data[i] = b._data[i];
  }
  FUNC_DECL mat3(T *data) {
    for (int i = 0; i < 9; ++i) _data[i] = data[i];
  }
  FUNC_DECL mat3(T x1, T y1, T z1, T x2, T y2, T z2, T x3, T y3, T z3) { _data = {x1, y1, z1, x2, y2, z2, x3, y3, z3}; }
  FUNC_DECL vec3<T> &operator[](const int i) { return value[i]; }
  FUNC_DECL T &operator()(int row, int col) { return _data[row * 3 + col]; }
  FUNC_DECL T &at(int row, int col) { return _data[row * 4 + col]; }
  FUNC_DECL mat3<T> static I() { return {T(1), 0, 0, 0, T(1), 0, 0, 0, T(1)}; }

  FUNC_DECL vec3<T> mul(const vec3<T> &p) {
    return {
        p.x * _data[0] + p.y * _data[1] + p.z * _data[2],
        p.x * _data[3] + p.y * _data[4] + p.z * _data[5],
        p.x * _data[6] + p.y * _data[7] + p.z * _data[8],
    };
  }
  FUNC_DECL vec3<T> operator*(const vec3<T> &p) { return mul(p); }
};

template <typename T>
struct mat4 {
  union {
    struct {
      vec4<T> value[4];
    };
    T _data[16];
  };

  FUNC_DECL mat4() {
    for (int i = 0; i < 16; ++i) _data[i] = 0;
  }
  FUNC_DECL mat4(T *data) {
    for (int i = 0; i < 16; ++i) _data[i] = data[i];
  }
  FUNC_DECL mat4(T x1, T y1, T z1, T w1, T x2, T y2, T z2, T w2, T x3, T y3, T z3, T w3, T x4, T y4, T z4, T w4) {
    value[0] = {x1, y1, z1, w1};
    value[1] = {x2, y2, z2, w2};
    value[2] = {x3, y3, z3, w3};
    value[3] = {x4, y4, z4, w4};
  }
  FUNC_DECL mat4(const mat3<T> &R, const vec3<T> &t) {
    value[0] = {R._data[0], R._data[1], R._data[2], t.x};
    value[1] = {R._data[3], R._data[4], R._data[5], t.y};
    value[2] = {R._data[6], R._data[7], R._data[8], t.z};
    value[3] = {0, 0, 0, T(1)};
  }
  FUNC_DECL vec4<T> &operator[](const int &i) { return value[i]; }
  FUNC_DECL T &operator()(const int &row, const int &col) { return _data[row * 4 + col]; }
  FUNC_DECL vec4<T> &operator()(int row) { return {_data[row + 0], _data[row + 1], _data[row + 2], _data[row + 3]}; }
  FUNC_DECL T &at(int row, int col) { return _data[row * 4 + col]; }

  FUNC_DECL mat4<T> static I() { return {T(1), 0, 0, 0, 0, T(1), 0, 0, 0, 0, T(1), 0, 0, 0, 0, T(1)}; }
  FUNC_DECL mat4<T> operator*(const mat4<T> &B) {
    mat4<T> C;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          C._data[i * 4 + j] += _data[i * 4 + k] * B._data[k * 4 + j];
        }
      }
    }
    return C;
  }
};

}  // namespace OPS_3D