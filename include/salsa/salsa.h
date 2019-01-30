#pragma once

#include <Eigen/Core>
#include "geometry/xform.h"
#include "multirotor_sim/estimator_base.h"
#include "multirotor_sim/satellite.h"
#include "multirotor_sim/wsg84.h"

#include "factors/imu.h"
#include "factors/mocap.h"
#include "factors/xform.h"

#include "salsa/logger.h"

using namespace std;
using namespace Eigen;
using namespace xform;


template <int WindowSize>
class Salsa : public multirotor_sim::EstimatorBase
{
public:

    enum
    {
        N = WindowSize,
        N_IMU = N,
        N_MOCAP = N,
    };


    Salsa()
    {
        initState();
        initFactors();
        initSolver();
    }


    void initState()
    {
        x_idx_ = 0;
        for (int i = 0; i < N; i++)
        {
            x_.col(i) = Xformd::Identity().elements();
            problem_.AddParameterBlock(x_.data() + i*7, 7, new XformParamAD);

            v_.col(i).setZero();
            problem_.AddParameterBlock(v_.data() + i*3, 3);

            tau_.col(i).setZero();
            problem_.AddParameterBlock(tau_.data() + i*2, 2);
        }
        imu_bias_.setZero();
        dt_mocap_ = 0.0;
        t_[0] = 0;
    }


    void initFactors()
    {
        imu_idx_ = 0;
        imu_full_ = false;
        for (int i = 0; i < N_IMU; i++)
            imu_factors_.push_back(new ImuFactorAD(&imu_[i]));
        imu_[0].reset(t_[0], imu_bias_);

        mocap_full_ = false;
        mocap_idx_ = 0;
        for (int i = 0; i < N_MOCAP; i++)
            mocap_factors_.push_back(new MocapFactorAD(&mocap_[i]));
    }


    void initSolver()
    {
        options_.max_num_iterations = 100;
        options_.num_threads = 6;
        options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options_.minimizer_progress_to_stdout = false;
    }


    void solve()
    {
        ceres::Solve(options_, &problem_, &summary_);
    }


    void imuCallback(const double &t, const Vector6d &z, const Matrix6d &R) override
    {
        imu_[imu_idx_].integrate(t, z, R);
    }


    void finishNode(const double& t)
    {
        int next_imu_idx = (imu_idx_ + 1) % N_IMU;
        int next_x_idx = (x_idx_ + 1) % N;

        if (next_imu_idx == 0)
            imu_full_ = true;

        if (imu_full_)
            problem_.RemoveResidualBlock(imu_res_ids_[next_imu_idx]);

        imu_[imu_idx_].integrate(t, imu_[imu_idx_].u_, imu_[imu_idx_].cov_);
        imu_[imu_idx_].finished();

        // Best Guess of next state
        imu_[imu_idx_].estimateXj(x_.data() + 7*x_idx_, v_.data() + 3*x_idx_, x_.data() + 7*next_x_idx, v_.data() + 3*next_x_idx);
        tau_(0, next_x_idx) = tau_(0, imu_idx_) + imu_[imu_idx_].delta_t_ * tau_(1, imu_idx_);
        tau_(1, next_x_idx) = tau_(1, imu_idx_);
        t_(next_x_idx) = t;



        imu_res_ids_[imu_idx_] = problem_.AddResidualBlock(imu_factors_[imu_idx_],
                                  NULL,
                                  x_.data() + 7*x_idx_,
                                  x_.data() + 7*next_x_idx,
                                  v_.data() + 3*x_idx_,
                                  v_.data() + 3*next_x_idx,
                                  imu_bias_.data());

        // Prepare the next imu factor
        imu_[next_imu_idx].reset(t, imu_bias_);
        imu_idx_ = next_imu_idx;
        x_idx_ = next_x_idx;
    }




    void mocapCallback(const double &t, const Xformd &z, const Matrix6d &R) override
    {
        int next_mocap_idx = (mocap_idx_ + 1) % N_MOCAP;
        int prev_x_idx = x_idx_;

        finishNode(t);

        if (next_mocap_idx == 0)
            mocap_full_ = true;

        if (mocap_full_)
            problem_.RemoveResidualBlock(mocap_res_ids_[next_mocap_idx]);

        x_.col(x_idx_) = z.elements();

        Vector6d zdot = (Xformd(x_.col(x_idx_)) - Xformd(x_.col(prev_x_idx))) / (t - t_[prev_x_idx]);

        mocap_[mocap_idx_].init(z.arr(), zdot, R);
        mocap_res_ids_[next_mocap_idx] = problem_.AddResidualBlock(mocap_factors_[mocap_idx_],
                                  NULL,
                                  x_.data() + 7*x_idx_,
                                  &dt_mocap_);


        mocap_idx_ = next_mocap_idx;
    }


    void featCallback(const double& t, const Vector2d& z, const Matrix2d& R, int id, double depth) override {}
    void rawGnssCallback(const GTime& t, const Vector3d& z, const Matrix3d& R, Satellite& sat) override { }

    Matrix<double, N, 1> t_;
    Matrix<double, 7, N> x_; int x_idx_;
    Matrix<double, 3, N> v_;
    Matrix<double, 2, N> tau_;
    Vector6d imu_bias_;
    double dt_mocap_;

    bool imu_full_;
    ImuFunctor imu_[N_IMU]; int imu_idx_;
    std::vector<ImuFactorAD*> imu_factors_;
    ceres::ResidualBlockId imu_res_ids_[N];

    bool mocap_full_;
    MocapFunctor mocap_[N]; int mocap_idx_;
    std::vector<MocapFactorAD*> mocap_factors_;
    ceres::ResidualBlockId mocap_res_ids_[N];

    ceres::Problem problem_;
    ceres::Solver::Options options_;
    ceres::Solver::Summary summary_;
};
