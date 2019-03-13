#include "factors/feat.h"

FeatFunctor::FeatFunctor(const Xformd& x_b2c, const Matrix2d& cov, const Camera<double>& cam) :
  cam_(cam),
  x_b2c_(x_b2c)
{
  xi_ = cov.inverse().llt().matrixL().transpose();
  active_ = false;
}

void FeatFunctor::init(const Vector2d &p1, const Vector2d &p2)
{
  p1_ = p1;
  p2_ = p2;
  active_ = true;
}

template <typename T>
bool FeatFunctor::operator ()(const T* _xi, const T* _xj, const T* _rho0, T* _res) const
{

}




