#pragma once
#define GLM_FORCE_QUAT_DATA_XYZW
#include <glm/glm.hpp>

template <typename T>
struct GLMTypes {
  using vec3 = glm::fvec3;
  using vec4 = glm::fvec4;
  using mat3 = glm::fmat3;
  using mat4 = glm::fmat4;
  static T t;
};

template <>
struct GLMTypes<float> {
  using vec3 = glm::fvec3;
  using vec4 = glm::fvec4;
  using mat3 = glm::fmat3;
  using mat4 = glm::fmat4;
};

template <>
struct GLMTypes<double> {
  using vec3 = glm::dvec3;
  using vec4 = glm::dvec4;
  using mat3 = glm::dmat3;
  using mat4 = glm::dmat4;
};

#define TM3 typename GLMTypes<T>::mat3
#define TM4 typename GLMTypes<T>::mat4

template <typename T>
void mat4_copy_to(TM4 &m, T *p) {
  p[0]  = m[0][0];
  p[1]  = m[1][0];
  p[2]  = m[2][0];
  p[3]  = m[3][0];
  p[4]  = m[0][1];
  p[5]  = m[1][1];
  p[6]  = m[2][1];
  p[7]  = m[3][1];
  p[8]  = m[0][2];
  p[9]  = m[1][2];
  p[10] = m[2][2];
  p[11] = m[3][2];
  p[12] = m[0][3];
  p[13] = m[1][3];
  p[14] = m[2][3];
  p[15] = m[3][3];
}

template <typename T>
void mat3_copy_to(TM3 &m, T *p) {
  p[0] = m[0][0];
  p[1] = m[1][0];
  p[2] = m[2][0];
  p[3] = m[0][1];
  p[4] = m[1][1];
  p[5] = m[2][1];
  p[6] = m[0][2];
  p[7] = m[1][2];
  p[8] = m[2][2];
}
