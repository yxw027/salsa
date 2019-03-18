#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "factors/shield.h"

#include "salsa/misc.h"
#include "geometry/xform.h"
#include "geometry/cam.h"

using namespace Eigen;
using namespace xform;

class FeatFunctor
{
public:
    FeatFunctor(const Xformd& x_b2c, const Matrix2d& cov,
                const Vector3d &zetai_, Vector3d &zetaj_);

    template<typename T>
    bool operator() (const T* _xi, const T* _xj, const T* _rho, T* _res) const;

    Vector3d zetai_, zetaj_;
    const Xformd& x_b2c_;
    Matrix2d Xi_;
    Matrix<double, 2, 3> Pz_;

    Matrix3d R_b2c;
};
