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
        FeatFunctor(const Xformd& x_b2c, const Matrix2d& cov, const Camera<double>& cam);
        void init(const Vector2d& p1, const Vector2d& p2);

        template<typename T>
        bool operator() (const T* _xi, const T* _xj, const T* _rho, T* _res) const;

        bool active_;
        Vector2d zeta1_, p2_, p1_;
        const Camera<double>& cam_;
        const Xformd& x_b2c_;
        Matrix2d xi_;
};
