#include "factors/switch_dynamics.h"

using namespace Eigen;

namespace salsa {


SwitchFunctor::SwitchFunctor(const double Xi)
{
    Xi_ = Xi;
}


template <typename T>
bool SwitchFunctor::operator()(const T* _si, const T* _sj, T* _res) const
{
    const T &si(*_si);
    const T &sj(*_sj);
    T &r(*_res);

    r = Xi_ * (si - sj);

    return true;
}
template bool SwitchFunctor::operator()<double>(const double* _si, const double* _sj, double* _res) const;
typedef ceres::Jet<double, 2> jactype;
template bool SwitchFunctor::operator()<jactype>(const jactype* _si, const jactype* _sj, jactype* _res) const;


SwitchFactor::SwitchFactor(double Xi)
{
    Xi_ = Xi;
}

bool SwitchFactor::Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
{
    const double &si(*parameters[0]);
    const double &sj(*parameters[1]);
    double &r(*residuals);

    r = Xi_ * (si - sj);

    if (jacobians)
    {
        if (jacobians[0])
        {
            double& drdsi(*jacobians[0]);
            drdsi = Xi_;
        }
        if (jacobians[1])
        {
            double& drdsj(*jacobians[1]);
            drdsj = -Xi_;
        }
    }
    return true;
}
}
