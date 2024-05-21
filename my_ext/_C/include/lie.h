/*
reference: https://github.com/princeton-vl/lietorch/
*/
#pragma once

#include <stdio.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "common.hpp"
namespace Lie {

#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#endif

#ifndef EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_RUNTIME_NO_MALLOC
#endif

#define EPS 1e-6
#define PI 3.14159265358979323846

template <typename Scalar>
class SO3 {
 public:
  const static int constexpr K = 3;  // manifold dimension
  const static int constexpr N = 4;  // embedding dimension

  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;

  using Tangent = Eigen::Matrix<Scalar, K, 1>;
  using Data    = Eigen::Matrix<Scalar, N, 1>;

  using Point          = Eigen::Matrix<Scalar, 3, 1>;
  using Point4         = Eigen::Matrix<Scalar, 4, 1>;
  using Transformation = Eigen::Matrix<Scalar, 3, 3>;
  using Adjoint        = Eigen::Matrix<Scalar, K, K>;
  using Quaternion     = Eigen::Quaternion<Scalar>;

  EIGEN_DEVICE_FUNC SO3(Quaternion const& q) : unit_quaternion(q) { unit_quaternion.normalize(); };

  EIGEN_DEVICE_FUNC SO3(const Scalar* data) : unit_quaternion(data) { unit_quaternion.normalize(); };

  EIGEN_DEVICE_FUNC SO3() { unit_quaternion = Quaternion::Identity(); }

  EIGEN_DEVICE_FUNC SO3<Scalar> inv() { return SO3<Scalar>(unit_quaternion.conjugate()); }

  EIGEN_DEVICE_FUNC Data data() const { return unit_quaternion.coeffs(); }

  EIGEN_DEVICE_FUNC SO3<Scalar> operator*(SO3<Scalar> const& other) {
    return SO3(unit_quaternion * other.unit_quaternion);
  }

  EIGEN_DEVICE_FUNC Point operator*(Point const& p) const {
    const Quaternion& q = unit_quaternion;
    Point uv            = q.vec().cross(p);
    uv += uv;
    return p + q.w() * uv + q.vec().cross(uv);
  }

  EIGEN_DEVICE_FUNC Point4 act4(Point4 const& p) const {
    Point4 p1;
    p1 << this->operator*(p.template segment<3>(0)), p(3);
    return p1;
  }

  EIGEN_DEVICE_FUNC Adjoint Adj() const { return unit_quaternion.toRotationMatrix(); }

  EIGEN_DEVICE_FUNC Transformation Matrix() const { return unit_quaternion.toRotationMatrix(); }

  EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 4, 4> Matrix4x4() const {
    Eigen::Matrix<Scalar, 4, 4> T = Eigen::Matrix<Scalar, 4, 4>::Identity();
    T.template block<3, 3>(0, 0)  = Matrix();
    return T;
  }

  EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 4, 4> orthogonal_projector() const {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 4, 4> J = Eigen::Matrix<Scalar, 4, 4>::Zero();
    J.template block<3, 3>(0, 0) =
        0.5 * (unit_quaternion.w() * Matrix3::Identity() + SO3<Scalar>::hat(-unit_quaternion.vec()));

    J.template block<1, 3>(3, 0) = 0.5 * (-unit_quaternion.vec());
    return J;
  }

  EIGEN_DEVICE_FUNC Tangent Adj(Tangent const& a) const { return Adj() * a; }

  EIGEN_DEVICE_FUNC Tangent AdjT(Tangent const& a) const { return Adj().transpose() * a; }

  EIGEN_DEVICE_FUNC static Transformation hat(Tangent const& phi) {
    Transformation Phi;
    Phi << 0.0, -phi(2), phi(1), phi(2), 0.0, -phi(0), -phi(1), phi(0), 0.0;

    return Phi;
  }

  EIGEN_DEVICE_FUNC static Adjoint adj(Tangent const& phi) { return SO3<Scalar>::hat(phi); }

  EIGEN_DEVICE_FUNC Tangent Log() const {
    using std::abs;
    using std::atan;
    using std::sqrt;
    Scalar squared_n = unit_quaternion.vec().squaredNorm();
    Scalar w         = unit_quaternion.w();

    Scalar two_atan_nbyw_by_n;

    /// Atan-based log thanks to
    ///
    /// C. Hertzberg et al.:
    /// "Integrating Generic Sensor Fusion Algorithms with Sound State
    /// Representation through Encapsulation of Manifolds"
    /// Information Fusion, 2011

    if (squared_n < EPS * EPS) {
      // If quaternion is normalized and n=0, then w should be 1;
      // w=0 should never happen here!
      Scalar squared_w   = w * w;
      two_atan_nbyw_by_n = Scalar(2) / w - Scalar(2.0 / 3.0) * (squared_n) / (w * squared_w);
    } else {
      Scalar n = sqrt(squared_n);
      if (abs(w) < EPS) {
        if (w > Scalar(0)) {
          two_atan_nbyw_by_n = Scalar(PI) / n;
        } else {
          two_atan_nbyw_by_n = -Scalar(PI) / n;
        }
      } else {
        two_atan_nbyw_by_n = Scalar(2) * atan(n / w) / n;
      }
    }

    return two_atan_nbyw_by_n * unit_quaternion.vec();
  }

  EIGEN_DEVICE_FUNC static SO3<Scalar> Exp(Tangent const& phi) {
    Scalar theta2 = phi.squaredNorm();
    Scalar theta  = sqrt(theta2);
    Scalar imag_factor;
    Scalar real_factor;

    if (theta < EPS) {
      Scalar theta4 = theta2 * theta2;
      imag_factor   = Scalar(0.5) - Scalar(1.0 / 48.0) * theta2 + Scalar(1.0 / 3840.0) * theta4;
      real_factor   = Scalar(1) - Scalar(1.0 / 8.0) * theta2 + Scalar(1.0 / 384.0) * theta4;
    } else {
      imag_factor = sin(.5 * theta) / theta;
      real_factor = cos(.5 * theta);
    }

    Quaternion q(real_factor, imag_factor * phi.x(), imag_factor * phi.y(), imag_factor * phi.z());
    return SO3<Scalar>(q);
  }

  EIGEN_DEVICE_FUNC static Adjoint left_jacobian(Tangent const& phi) {
    // left jacobian
    Matrix3 I    = Matrix3::Identity();
    Matrix3 Phi  = SO3<Scalar>::hat(phi);
    Matrix3 Phi2 = Phi * Phi;

    Scalar theta2 = phi.squaredNorm();
    Scalar theta  = sqrt(theta2);

    Scalar coef1 = (theta < EPS) ? Scalar(1.0 / 2.0) - Scalar(1.0 / 24.0) * theta2 : (1.0 - cos(theta)) / theta2;

    Scalar coef2 =
        (theta < EPS) ? Scalar(1.0 / 6.0) - Scalar(1.0 / 120.0) * theta2 : (theta - sin(theta)) / (theta2 * theta);

    return I + coef1 * Phi + coef2 * Phi2;
  }

  EIGEN_DEVICE_FUNC static Adjoint left_jacobian_inverse(Tangent const& phi) {
    // left jacobian inverse
    Matrix3 I    = Matrix3::Identity();
    Matrix3 Phi  = SO3<Scalar>::hat(phi);
    Matrix3 Phi2 = Phi * Phi;

    Scalar theta2     = phi.squaredNorm();
    Scalar theta      = sqrt(theta2);
    Scalar half_theta = Scalar(.5) * theta;

    Scalar coef2 = (theta < EPS)
                       ? Scalar(1.0 / 12.0)
                       : (Scalar(1) - theta * cos(half_theta) / (Scalar(2) * sin(half_theta))) / (theta * theta);

    return I + Scalar(-0.5) * Phi + coef2 * Phi2;
  }

  EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar, 3, 3> act_jacobian(Point const& p) {
    // jacobian action on a point
    return SO3<Scalar>::hat(-p);
  }

  EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar, 4, 3> act4_jacobian(Point4 const& p) {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 4, 3> J = Eigen::Matrix<Scalar, 4, 3>::Zero();
    J.template block<3, 3>(0, 0)  = SO3<Scalar>::hat(-p.template segment<3>(0));
    return J;
  }

 private:
  Quaternion unit_quaternion;
};

template <typename Scalar>
class SE3 {
 public:
  const static int constexpr K = 6;  // manifold dimension
  const static int constexpr N = 7;  // embedding dimension

  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;

  using Tangent        = Eigen::Matrix<Scalar, K, 1>;
  using Point          = Eigen::Matrix<Scalar, 3, 1>;
  using Point4         = Eigen::Matrix<Scalar, 4, 1>;
  using Data           = Eigen::Matrix<Scalar, N, 1>;
  using Transformation = Eigen::Matrix<Scalar, 4, 4>;
  using Adjoint        = Eigen::Matrix<Scalar, K, K>;

  EIGEN_DEVICE_FUNC SE3() { translation = Vector3::Zero(); }

  EIGEN_DEVICE_FUNC SE3(SO3<Scalar> const& so3, Vector3 const& t) : so3(so3), translation(t){};

  EIGEN_DEVICE_FUNC SE3(const Scalar* data) : translation(data), so3(data + 3){};

  EIGEN_DEVICE_FUNC SE3<Scalar> inv() { return SE3(so3.inv(), -(so3.inv() * translation)); }

  EIGEN_DEVICE_FUNC Data data() const {
    Data data_vec;
    data_vec << translation, so3.data();
    return data_vec;
  }

  EIGEN_DEVICE_FUNC SE3<Scalar> operator*(SE3<Scalar> const& other) {
    return SE3(so3 * other.so3, translation + so3 * other.translation);
  }

  EIGEN_DEVICE_FUNC Point operator*(Point const& p) const { return so3 * p + translation; }

  EIGEN_DEVICE_FUNC Point4 act4(Point4 const& p) const {
    Point4 p1;
    p1 << so3 * p.template segment<3>(0) + translation * p(3), p(3);
    return p1;
  }

  EIGEN_DEVICE_FUNC Adjoint Adj() const {
    Matrix3 R   = so3.Matrix();
    Matrix3 tx  = SO3<Scalar>::hat(translation);
    Matrix3 Zer = Matrix3::Zero();

    Adjoint Ad;
    Ad << R, tx * R, Zer, R;

    return Ad;
  }

  EIGEN_DEVICE_FUNC Transformation Matrix() const {
    Transformation T             = Transformation::Identity();
    T.template block<3, 3>(0, 0) = so3.Matrix();
    T.template block<3, 1>(0, 3) = translation;
    return T;
  }

  EIGEN_DEVICE_FUNC Transformation Matrix4x4() const { return Matrix(); }

  EIGEN_DEVICE_FUNC Tangent Adj(Tangent const& a) const { return Adj() * a; }

  EIGEN_DEVICE_FUNC Tangent AdjT(Tangent const& a) const { return Adj().transpose() * a; }

  EIGEN_DEVICE_FUNC static Transformation hat(Tangent const& tau_phi) {
    Vector3 tau = tau_phi.template segment<3>(0);
    Vector3 phi = tau_phi.template segment<3>(3);

    Transformation TauPhi             = Transformation::Zero();
    TauPhi.template block<3, 3>(0, 0) = SO3<Scalar>::hat(phi);
    TauPhi.template block<3, 1>(0, 3) = tau;

    return TauPhi;
  }

  EIGEN_DEVICE_FUNC static Adjoint adj(Tangent const& tau_phi) {
    Vector3 tau = tau_phi.template segment<3>(0);
    Vector3 phi = tau_phi.template segment<3>(3);

    Matrix3 Tau = SO3<Scalar>::hat(tau);
    Matrix3 Phi = SO3<Scalar>::hat(phi);
    Matrix3 Zer = Matrix3::Zero();

    Adjoint ad;
    ad << Phi, Tau, Zer, Phi;

    return ad;
  }

  EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 7, 7> orthogonal_projector() const {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 7, 7> J = Eigen::Matrix<Scalar, 7, 7>::Zero();
    J.template block<3, 3>(0, 0)  = Matrix3::Identity();
    J.template block<3, 3>(0, 3)  = SO3<Scalar>::hat(-translation);
    J.template block<4, 4>(3, 3)  = so3.orthogonal_projector();

    return J;
  }

  EIGEN_DEVICE_FUNC Tangent Log() const {
    Vector3 phi  = so3.Log();
    Matrix3 Vinv = SO3<Scalar>::left_jacobian_inverse(phi);

    Tangent tau_phi;
    tau_phi << Vinv * translation, phi;

    return tau_phi;
  }

  EIGEN_DEVICE_FUNC static SE3<Scalar> Exp(Tangent const& tau_phi) {
    Vector3 tau = tau_phi.template segment<3>(0);
    Vector3 phi = tau_phi.template segment<3>(3);

    SO3<Scalar> so3 = SO3<Scalar>::Exp(phi);
    Vector3 t       = SO3<Scalar>::left_jacobian(phi) * tau;

    return SE3<Scalar>(so3, t);
  }

  EIGEN_DEVICE_FUNC static Matrix3 calcQ(Tangent const& tau_phi) {
    // Q matrix
    Vector3 tau = tau_phi.template segment<3>(0);
    Vector3 phi = tau_phi.template segment<3>(3);
    Matrix3 Tau = SO3<Scalar>::hat(tau);
    Matrix3 Phi = SO3<Scalar>::hat(phi);

    Scalar theta      = phi.norm();
    Scalar theta_pow2 = theta * theta;
    Scalar theta_pow4 = theta_pow2 * theta_pow2;

    Scalar coef1 = (theta < EPS) ? Scalar(1.0 / 6.0) - Scalar(1.0 / 120.0) * theta_pow2
                                 : (theta - sin(theta)) / (theta_pow2 * theta);

    Scalar coef2 = (theta < EPS) ? Scalar(1.0 / 24.0) - Scalar(1.0 / 720.0) * theta_pow2
                                 : (theta_pow2 + 2 * cos(theta) - 2) / (2 * theta_pow4);

    Scalar coef3 = (theta < EPS) ? Scalar(1.0 / 120.0) - Scalar(1.0 / 2520.0) * theta_pow2
                                 : (2 * theta - 3 * sin(theta) + theta * cos(theta)) / (2 * theta_pow4 * theta);

    Matrix3 Q = Scalar(0.5) * Tau + coef1 * (Phi * Tau + Tau * Phi + Phi * Tau * Phi) +
                coef2 * (Phi * Phi * Tau + Tau * Phi * Phi - 3 * Phi * Tau * Phi) +
                coef3 * (Phi * Tau * Phi * Phi + Phi * Phi * Tau * Phi);

    return Q;
  }

  EIGEN_DEVICE_FUNC static Adjoint left_jacobian(Tangent const& tau_phi) {
    // left jacobian
    Vector3 phi = tau_phi.template segment<3>(3);
    Matrix3 J   = SO3<Scalar>::left_jacobian(phi);
    Matrix3 Q   = SE3<Scalar>::calcQ(tau_phi);
    Matrix3 Zer = Matrix3::Zero();

    Adjoint J6x6;
    J6x6 << J, Q, Zer, J;

    return J6x6;
  }

  EIGEN_DEVICE_FUNC static Adjoint left_jacobian_inverse(Tangent const& tau_phi) {
    // left jacobian inverse
    Vector3 tau  = tau_phi.template segment<3>(0);
    Vector3 phi  = tau_phi.template segment<3>(3);
    Matrix3 Jinv = SO3<Scalar>::left_jacobian_inverse(phi);
    Matrix3 Q    = SE3<Scalar>::calcQ(tau_phi);
    Matrix3 Zer  = Matrix3::Zero();

    Adjoint J6x6;
    J6x6 << Jinv, -Jinv * Q * Jinv, Zer, Jinv;

    return J6x6;
  }

  EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar, 3, 6> act_jacobian(Point const& p) {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 3, 6> J;
    J.template block<3, 3>(0, 0) = Matrix3::Identity();
    J.template block<3, 3>(0, 3) = SO3<Scalar>::hat(-p);
    return J;
  }

  EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar, 4, 6> act4_jacobian(Point4 const& p) {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 4, 6> J = Eigen::Matrix<Scalar, 4, 6>::Zero();
    J.template block<3, 3>(0, 0)  = p(3) * Matrix3::Identity();
    J.template block<3, 3>(0, 3)  = SO3<Scalar>::hat(-p.template segment<3>(0));
    return J;
  }

 private:
  SO3<Scalar> so3;
  Vector3 translation;
};

template <typename Scalar>
class RxSO3 {
 public:
  const static int constexpr K = 4;  // manifold dimension
  const static int constexpr N = 5;  // embedding dimension

  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;

  using Tangent = Eigen::Matrix<Scalar, K, 1>;
  using Data    = Eigen::Matrix<Scalar, N, 1>;

  using Point  = Eigen::Matrix<Scalar, 3, 1>;
  using Point4 = Eigen::Matrix<Scalar, 4, 1>;

  using Quaternion     = Eigen::Quaternion<Scalar>;
  using Transformation = Eigen::Matrix<Scalar, 3, 3>;
  using Adjoint        = Eigen::Matrix<Scalar, 4, 4>;

  EIGEN_DEVICE_FUNC RxSO3(Quaternion const& q, Scalar const s) : unit_quaternion(q), scale(s) {
    unit_quaternion.normalize();
  };

  EIGEN_DEVICE_FUNC RxSO3(const Scalar* data) : unit_quaternion(data), scale(data[4]) { unit_quaternion.normalize(); };

  EIGEN_DEVICE_FUNC RxSO3() {
    unit_quaternion = Quaternion::Identity();
    scale           = Scalar(1.0);
  }

  EIGEN_DEVICE_FUNC RxSO3<Scalar> inv() { return RxSO3<Scalar>(unit_quaternion.conjugate(), 1.0 / scale); }

  EIGEN_DEVICE_FUNC Data data() const {
    Data data_vec;
    data_vec << unit_quaternion.coeffs(), scale;
    return data_vec;
  }

  EIGEN_DEVICE_FUNC RxSO3<Scalar> operator*(RxSO3<Scalar> const& other) {
    return RxSO3<Scalar>(unit_quaternion * other.unit_quaternion, scale * other.scale);
  }

  EIGEN_DEVICE_FUNC Point operator*(Point const& p) const {
    const Quaternion& q = unit_quaternion;
    Point uv            = q.vec().cross(p);
    uv += uv;
    return scale * (p + q.w() * uv + q.vec().cross(uv));
  }

  EIGEN_DEVICE_FUNC Point4 act4(Point4 const& p) const {
    Point4 p1;
    p1 << this->operator*(p.template segment<3>(0)), p(3);
    return p1;
  }

  EIGEN_DEVICE_FUNC Adjoint Adj() const {
    Adjoint Ad                    = Adjoint::Identity();
    Ad.template block<3, 3>(0, 0) = unit_quaternion.toRotationMatrix();
    return Ad;
  }

  EIGEN_DEVICE_FUNC Transformation Matrix() const { return scale * unit_quaternion.toRotationMatrix(); }

  EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 4, 4> Matrix4x4() const {
    Eigen::Matrix<Scalar, 4, 4> T;
    T                            = Eigen::Matrix<Scalar, 4, 4>::Identity();
    T.template block<3, 3>(0, 0) = Matrix();
    return T;
  }

  EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 5, 5> orthogonal_projector() const {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 5, 5> J = Eigen::Matrix<Scalar, 5, 5>::Zero();

    J.template block<3, 3>(0, 0) =
        0.5 * (unit_quaternion.w() * Matrix3::Identity() + SO3<Scalar>::hat(-unit_quaternion.vec()));

    J.template block<1, 3>(3, 0) = 0.5 * (-unit_quaternion.vec());

    // scale
    J(4, 3) = scale;

    return J;
  }

  EIGEN_DEVICE_FUNC Transformation Rotation() const { return unit_quaternion.toRotationMatrix(); }

  EIGEN_DEVICE_FUNC Tangent Adj(Tangent const& a) const { return Adj() * a; }

  EIGEN_DEVICE_FUNC Tangent AdjT(Tangent const& a) const { return Adj().transpose() * a; }

  EIGEN_DEVICE_FUNC static Transformation hat(Tangent const& phi_sigma) {
    Vector3 const phi = phi_sigma.template segment<3>(0);
    return SO3<Scalar>::hat(phi) + phi(3) * Transformation::Identity();
  }

  EIGEN_DEVICE_FUNC static Adjoint adj(Tangent const& phi_sigma) {
    Vector3 const phi = phi_sigma.template segment<3>(0);
    Matrix3 const Phi = SO3<Scalar>::hat(phi);

    Adjoint ad                    = Adjoint::Zero();
    ad.template block<3, 3>(0, 0) = Phi;

    return ad;
  }

  EIGEN_DEVICE_FUNC Tangent Log() const {
    using std::abs;
    using std::atan;
    using std::sqrt;

    Scalar squared_n = unit_quaternion.vec().squaredNorm();
    Scalar w         = unit_quaternion.w();
    Scalar two_atan_nbyw_by_n;

    /// Atan-based log thanks to
    ///
    /// C. Hertzberg et al.:
    /// "Integrating Generic Sensor Fusion Algorithms with Sound State
    /// Representation through Encapsulation of Manifolds"
    /// Information Fusion, 2011

    if (squared_n < EPS * EPS) {
      two_atan_nbyw_by_n = Scalar(2) / w - Scalar(2.0 / 3.0) * (squared_n) / (w * w * w);
    } else {
      Scalar n = sqrt(squared_n);
      if (abs(w) < EPS) {
        if (w > Scalar(0)) {
          two_atan_nbyw_by_n = PI / n;
        } else {
          two_atan_nbyw_by_n = -PI / n;
        }
      } else {
        two_atan_nbyw_by_n = Scalar(2) * atan(n / w) / n;
      }
    }

    Tangent phi_sigma;
    phi_sigma << two_atan_nbyw_by_n * unit_quaternion.vec(), log(scale);

    return phi_sigma;
  }

  EIGEN_DEVICE_FUNC static RxSO3<Scalar> Exp(Tangent const& phi_sigma) {
    Vector3 phi  = phi_sigma.template segment<3>(0);
    Scalar scale = exp(phi_sigma(3));

    Scalar theta2 = phi.squaredNorm();
    Scalar theta  = sqrt(theta2);
    Scalar imag_factor;
    Scalar real_factor;

    if (theta < EPS) {
      Scalar theta4 = theta2 * theta2;
      imag_factor   = Scalar(0.5) - Scalar(1.0 / 48.0) * theta2 + Scalar(1.0 / 3840.0) * theta4;
      real_factor   = Scalar(1) - Scalar(1.0 / 8.0) * theta2 + Scalar(1.0 / 384.0) * theta4;
    } else {
      imag_factor = sin(.5 * theta) / theta;
      real_factor = cos(.5 * theta);
    }

    Quaternion q(real_factor, imag_factor * phi.x(), imag_factor * phi.y(), imag_factor * phi.z());
    return RxSO3<Scalar>(q, scale);
  }

  EIGEN_DEVICE_FUNC static Matrix3 calcW(Tangent const& phi_sigma) {
    // left jacobian
    using std::abs;
    Matrix3 const I = Matrix3::Identity();
    Scalar const one(1);
    Scalar const half(0.5);

    Vector3 const phi  = phi_sigma.template segment<3>(0);
    Scalar const sigma = phi_sigma(3);
    Scalar const theta = phi.norm();

    Matrix3 const Phi  = SO3<Scalar>::hat(phi);
    Matrix3 const Phi2 = Phi * Phi;
    Scalar const scale = exp(sigma);

    Scalar A, B, C;
    if (abs(sigma) < EPS) {
      C = one;
      if (abs(theta) < EPS) {
        A = half;
        B = Scalar(1. / 6.);
      } else {
        Scalar theta_sq = theta * theta;
        A               = (one - cos(theta)) / theta_sq;
        B               = (theta - sin(theta)) / (theta_sq * theta);
      }
    } else {
      C = (scale - one) / sigma;
      if (abs(theta) < EPS) {
        Scalar sigma_sq = sigma * sigma;
        A               = ((sigma - one) * scale + one) / sigma_sq;
        B               = (scale * half * sigma_sq + scale - one - sigma * scale) / (sigma_sq * sigma);
      } else {
        Scalar theta_sq = theta * theta;
        Scalar a        = scale * sin(theta);
        Scalar b        = scale * cos(theta);
        Scalar c        = theta_sq + sigma * sigma;
        A               = (a * sigma + (one - b) * theta) / (theta * c);
        B               = (C - ((b - one) * sigma + a * theta) / (c)) * one / (theta_sq);
      }
    }
    return A * Phi + B * Phi2 + C * I;
  }

  EIGEN_DEVICE_FUNC static Matrix3 calcWInv(Tangent const& phi_sigma) {
    // left jacobian inverse
    Matrix3 const I = Matrix3::Identity();
    Scalar const half(0.5);
    Scalar const one(1);
    Scalar const two(2);

    Vector3 const phi  = phi_sigma.template segment<3>(0);
    Scalar const sigma = phi_sigma(3);
    Scalar const theta = phi.norm();
    Scalar const scale = exp(sigma);

    Matrix3 const Phi      = SO3<Scalar>::hat(phi);
    Matrix3 const Phi2     = Phi * Phi;
    Scalar const scale_sq  = scale * scale;
    Scalar const theta_sq  = theta * theta;
    Scalar const sin_theta = sin(theta);
    Scalar const cos_theta = cos(theta);

    Scalar a, b, c;
    if (abs(sigma * sigma) < EPS) {
      c = one - half * sigma;
      a = -half;
      if (abs(theta_sq) < EPS) {
        b = Scalar(1. / 12.);
      } else {
        b = (theta * sin_theta + two * cos_theta - two) / (two * theta_sq * (cos_theta - one));
      }
    } else {
      Scalar const scale_cu = scale_sq * scale;
      c                     = sigma / (scale - one);
      if (abs(theta_sq) < EPS) {
        a = (-sigma * scale + scale - one) / ((scale - one) * (scale - one));
        b = (scale_sq * sigma - two * scale_sq + scale * sigma + two * scale) /
            (two * scale_cu - Scalar(6) * scale_sq + Scalar(6) * scale - two);
      } else {
        Scalar const s_sin_theta = scale * sin_theta;
        Scalar const s_cos_theta = scale * cos_theta;
        a = (theta * s_cos_theta - theta - sigma * s_sin_theta) / (theta * (scale_sq - two * s_cos_theta + one));
        b = -scale *
            (theta * s_sin_theta - theta * sin_theta + sigma * s_cos_theta - scale * sigma + sigma * cos_theta -
                sigma) /
            (theta_sq * (scale_cu - two * scale * s_cos_theta - scale_sq + two * s_cos_theta + scale - one));
      }
    }
    return a * Phi + b * Phi2 + c * I;
  }

  EIGEN_DEVICE_FUNC static Adjoint left_jacobian(Tangent const& phi_sigma) {
    // left jacobian
    Adjoint J                    = Adjoint::Identity();
    Vector3 phi                  = phi_sigma.template segment<3>(0);
    J.template block<3, 3>(0, 0) = SO3<Scalar>::left_jacobian(phi);
    return J;
  }

  EIGEN_DEVICE_FUNC static Adjoint left_jacobian_inverse(Tangent const& phi_sigma) {
    // left jacobian inverse
    Adjoint Jinv                    = Adjoint::Identity();
    Vector3 phi                     = phi_sigma.template segment<3>(0);
    Jinv.template block<3, 3>(0, 0) = SO3<Scalar>::left_jacobian_inverse(phi);
    return Jinv;
  }

  EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar, 3, 4> act_jacobian(Point const& p) {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 3, 4> Ja;
    Ja << SO3<Scalar>::hat(-p), p;
    return Ja;
  }

  EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar, 4, 4> act4_jacobian(Point4 const& p) {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 4, 4> J = Eigen::Matrix<Scalar, 4, 4>::Zero();
    J.template block<3, 3>(0, 0)  = SO3<Scalar>::hat(-p.template segment<3>(0));
    J.template block<3, 1>(0, 3)  = p.template segment<3>(0);
    return J;
  }

 private:
  Quaternion unit_quaternion;
  Scalar scale;
};

template <typename Scalar>
class Sim3 {
 public:
  const static int constexpr K = 7;  // manifold dimension
  const static int constexpr N = 8;  // embedding dimension

  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;

  using Tangent        = Eigen::Matrix<Scalar, K, 1>;
  using Point          = Eigen::Matrix<Scalar, 3, 1>;
  using Point4         = Eigen::Matrix<Scalar, 4, 1>;
  using Data           = Eigen::Matrix<Scalar, N, 1>;
  using Transformation = Eigen::Matrix<Scalar, 4, 4>;
  using Adjoint        = Eigen::Matrix<Scalar, K, K>;

  EIGEN_DEVICE_FUNC Sim3() { translation = Vector3::Zero(); }

  EIGEN_DEVICE_FUNC Sim3(RxSO3<Scalar> const& rxso3, Vector3 const& t) : rxso3(rxso3), translation(t){};

  EIGEN_DEVICE_FUNC Sim3(const Scalar* data) : translation(data), rxso3(data + 3){};

  EIGEN_DEVICE_FUNC Sim3<Scalar> inv() { return Sim3<Scalar>(rxso3.inv(), -(rxso3.inv() * translation)); }

  EIGEN_DEVICE_FUNC Data data() const {
    Data data_vec;
    data_vec << translation, rxso3.data();
    return data_vec;
  }

  EIGEN_DEVICE_FUNC Sim3<Scalar> operator*(Sim3<Scalar> const& other) {
    return Sim3(rxso3 * other.rxso3, translation + rxso3 * other.translation);
  }

  EIGEN_DEVICE_FUNC Point operator*(Point const& p) const { return (rxso3 * p) + translation; }

  EIGEN_DEVICE_FUNC Point4 act4(Point4 const& p) const {
    Point4 p1;
    p1 << rxso3 * p.template segment<3>(0) + p(3) * translation, p(3);
    return p1;
  }

  EIGEN_DEVICE_FUNC Transformation Matrix() const {
    Transformation T             = Transformation::Identity();
    T.template block<3, 3>(0, 0) = rxso3.Matrix();
    T.template block<3, 1>(0, 3) = translation;
    return T;
  }

  EIGEN_DEVICE_FUNC Transformation Matrix4x4() const {
    Transformation T             = Transformation::Identity();
    T.template block<3, 3>(0, 0) = rxso3.Matrix();
    T.template block<3, 1>(0, 3) = translation;
    return T;
  }

  EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 8, 8> orthogonal_projector() const {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 8, 8> J = Eigen::Matrix<Scalar, 8, 8>::Zero();
    J.template block<3, 3>(0, 0)  = Matrix3::Identity();
    J.template block<3, 3>(0, 3)  = SO3<Scalar>::hat(-translation);
    J.template block<3, 1>(0, 6)  = translation;
    J.template block<5, 5>(3, 3)  = rxso3.orthogonal_projector();
    return J;
  }

  EIGEN_DEVICE_FUNC Adjoint Adj() const {
    Adjoint Ad = Adjoint::Identity();
    Matrix3 sR = rxso3.Matrix();
    Matrix3 tx = SO3<Scalar>::hat(translation);
    Matrix3 R  = rxso3.Rotation();

    Ad.template block<3, 3>(0, 0) = sR;
    Ad.template block<3, 3>(0, 3) = tx * R;
    Ad.template block<3, 1>(0, 6) = -translation;
    Ad.template block<3, 3>(3, 3) = R;

    return Ad;
  }

  EIGEN_DEVICE_FUNC Tangent Adj(Tangent const& a) const { return Adj() * a; }

  EIGEN_DEVICE_FUNC Tangent AdjT(Tangent const& a) const { return Adj().transpose() * a; }

  EIGEN_DEVICE_FUNC static Transformation hat(Tangent const& tau_phi_sigma) {
    Vector3 tau  = tau_phi_sigma.template segment<3>(0);
    Vector3 phi  = tau_phi_sigma.template segment<3>(3);
    Scalar sigma = tau_phi_sigma(6);

    Matrix3 Phi = SO3<Scalar>::hat(phi);
    Matrix3 I   = Matrix3::Identity();

    Transformation Omega             = Transformation::Zero();
    Omega.template block<3, 3>(0, 0) = Phi + sigma * I;
    Omega.template block<3, 1>(0, 3) = tau;

    return Omega;
  }

  EIGEN_DEVICE_FUNC static Adjoint adj(Tangent const& tau_phi_sigma) {
    Adjoint ad   = Adjoint::Zero();
    Vector3 tau  = tau_phi_sigma.template segment<3>(0);
    Vector3 phi  = tau_phi_sigma.template segment<3>(3);
    Scalar sigma = tau_phi_sigma(6);

    Matrix3 Tau = SO3<Scalar>::hat(tau);
    Matrix3 Phi = SO3<Scalar>::hat(phi);
    Matrix3 I   = Matrix3::Identity();

    ad.template block<3, 3>(0, 0) = Phi + sigma * I;
    ad.template block<3, 3>(0, 3) = Tau;
    ad.template block<3, 1>(0, 6) = -tau;
    ad.template block<3, 3>(3, 3) = Phi;

    return ad;
  }

  EIGEN_DEVICE_FUNC Tangent Log() const {
    // logarithm map
    Vector4 phi_sigma = rxso3.Log();
    Matrix3 W         = RxSO3<Scalar>::calcW(phi_sigma);

    Tangent tau_phi_sigma;
    tau_phi_sigma << W.inverse() * translation, phi_sigma;

    return tau_phi_sigma;
  }

  EIGEN_DEVICE_FUNC static Sim3<Scalar> Exp(Tangent const& tau_phi_sigma) {
    // exponential map
    Vector3 tau       = tau_phi_sigma.template segment<3>(0);
    Vector4 phi_sigma = tau_phi_sigma.template segment<4>(3);

    RxSO3<Scalar> rxso3 = RxSO3<Scalar>::Exp(phi_sigma);
    Matrix3 W           = RxSO3<Scalar>::calcW(phi_sigma);

    return Sim3<Scalar>(rxso3, W * tau);
  }

  EIGEN_DEVICE_FUNC static Adjoint left_jacobian(Tangent const& tau_phi_sigma) {
    // left jacobian
    Adjoint const Xi  = adj(tau_phi_sigma);
    Adjoint const Xi2 = Xi * Xi;
    Adjoint const Xi4 = Xi2 * Xi2;

    return Adjoint::Identity() + Scalar(1.0 / 2.0) * Xi + Scalar(1.0 / 6.0) * Xi2 + Scalar(1.0 / 24.0) * Xi * Xi2 +
           Scalar(1.0 / 120.0) * Xi4;
    +Scalar(1.0 / 720.0) * Xi* Xi4;
  }

  EIGEN_DEVICE_FUNC static Adjoint left_jacobian_inverse(Tangent const& tau_phi_sigma) {
    // left jacobian inverse
    Adjoint const Xi  = adj(tau_phi_sigma);
    Adjoint const Xi2 = Xi * Xi;
    Adjoint const Xi4 = Xi2 * Xi2;

    return Adjoint::Identity() - Scalar(1.0 / 2.0) * Xi + Scalar(1.0 / 12.0) * Xi2 - Scalar(1.0 / 720.0) * Xi4;
  }

  EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar, 3, 7> act_jacobian(Point const& p) {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 3, 7> J;
    J.template block<3, 3>(0, 0) = Matrix3::Identity();
    J.template block<3, 3>(0, 3) = SO3<Scalar>::hat(-p);
    J.template block<3, 1>(0, 6) = p;
    return J;
  }

  EIGEN_DEVICE_FUNC static Eigen::Matrix<Scalar, 4, 7> act4_jacobian(Point4 const& p) {
    // jacobian action on a point
    Eigen::Matrix<Scalar, 4, 7> J = Eigen::Matrix<Scalar, 4, 7>::Zero();
    J.template block<3, 3>(0, 0)  = p(3) * Matrix3::Identity();
    J.template block<3, 3>(0, 3)  = SO3<Scalar>::hat(-p.template segment<3>(0));
    J.template block<3, 1>(0, 6)  = p.template segment<3>(0);
    return J;
  }

 private:
  Vector3 translation;
  RxSO3<Scalar> rxso3;
};

#define PRIVATE_CASE_TYPE(group_index, enum_type, type, ...) \
  case enum_type: {                                          \
    using scalar_t = type;                                   \
    switch (group_index) {                                   \
      case 1: {                                              \
        using group_t = SO3<type>;                           \
        return __VA_ARGS__();                                \
      }                                                      \
      case 2: {                                              \
        using group_t = RxSO3<type>;                         \
        return __VA_ARGS__();                                \
      }                                                      \
      case 3: {                                              \
        using group_t = SE3<type>;                           \
        return __VA_ARGS__();                                \
      }                                                      \
      case 4: {                                              \
        using group_t = Sim3<type>;                          \
        return __VA_ARGS__();                                \
      }                                                      \
    }                                                        \
  }

#define DISPATCH_GROUP_AND_FLOATING_TYPES(GROUP_INDEX, TYPE, NAME, ...)            \
  [&] {                                                                            \
    const auto& the_type = TYPE;                                                   \
    /* don't use TYPE again in case it is an expensive or side-effect op */        \
    at::ScalarType _st = ::detail::scalar_type(the_type);                          \
    switch (_st) {                                                                 \
      PRIVATE_CASE_TYPE(GROUP_INDEX, at::ScalarType::Double, double, __VA_ARGS__); \
      PRIVATE_CASE_TYPE(GROUP_INDEX, at::ScalarType::Float, float, __VA_ARGS__);   \
      default: break;                                                              \
    }                                                                              \
  }();

}  // namespace Lie