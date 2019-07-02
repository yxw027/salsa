#include "salsa/salsa.h"

using namespace std;
using namespace Eigen;
using namespace xform;

namespace salsa
{

void Salsa::initFactors()
{
    bias_ = new ImuBiasAnchor(imu_bias_, imu_bias_Xi_);
    state_anchor_ = new StateAnchor(state_anchor_xi_);
    x_e2n_anchor_ = new XformAnchor(x_e2n_anchor_xi_);
}

void Salsa::addParameterBlocks(ceres::Problem &problem)
{
    problem.AddParameterBlock(x_e2n_.data(), 7, new XformParam());
    if (!enable_static_start_ || xhead().t > static_start_end_)
        problem.SetParameterBlockConstant(x_e2n_.data());
    problem.AddParameterBlock(imu_bias_.data(), 6);

    int idx = xbuf_tail_;
    int prev_idx = -1;
    while (prev_idx != xbuf_head_)
    {
        SALSA_ASSERT(std::abs(1.0 - xbuf_[idx].x.q().arr_.norm()) < 1e-8, "Quat Left Manifold");
        problem.AddParameterBlock(xbuf_[idx].x.data(), 7, new XformParam());
        problem.AddParameterBlock(xbuf_[idx].v.data(), 3);
        problem.AddParameterBlock(xbuf_[idx].tau.data(), 2);
        prev_idx = idx;
        idx = (idx+1) % STATE_BUF_SIZE;
    }

    for (auto& feat : xfeat_)
    {
        Feat& ft(feat.second);
        if (ft.funcs.size() > 0)
        {
            problem.AddParameterBlock(&ft.rho, 1);
            problem.SetParameterLowerBound(&ft.rho, 0, 0.01);
        }
    }

    for (auto& pvec : prange_)
    {
        for (auto& p : pvec)
        {
            problem.AddParameterBlock(&p.sw, 1);
            problem.SetParameterLowerBound(&p.sw, 0, 0);
            problem.SetParameterUpperBound(&p.sw, 0, 1);
            if (!enable_switching_factors_)
                problem.SetParameterBlockConstant(&p.sw);
        }
    }
}

void Salsa::setAnchors(ceres::Problem &problem)
{
    if (xbuf_tail_ == xbuf_head_)
        return;

    x_e2n_anchor_->set(x_e2n_);
    FunctorShield<XformAnchor>* xe2n_ptr = new FunctorShield<XformAnchor>(x_e2n_anchor_);
    problem.AddResidualBlock(new XformAnchorFactorAD(xe2n_ptr), NULL, x_e2n_.data());

    state_anchor_->set(xbuf_[xbuf_tail_]);
    FunctorShield<StateAnchor>* state_ptr = new FunctorShield<StateAnchor>(state_anchor_);
    problem.AddResidualBlock(new StateAnchorFactorAD(state_ptr), NULL, xbuf_[xbuf_tail_].x.data(),
                             xbuf_[xbuf_tail_].v.data(), xbuf_[xbuf_tail_].tau.data());

    bias_->setBias(imu_bias_);
    FunctorShield<ImuBiasAnchor>* imu_ptr = new FunctorShield<ImuBiasAnchor>(bias_);
    problem.AddResidualBlock(new ImuBiasAnchorFactorAD(imu_ptr), NULL, imu_bias_.data());
}

void Salsa::addImuFactors(ceres::Problem &problem)
{
    int prev_idx = xbuf_tail_;
    for (auto it = imu_.begin(); it != imu_.end(); it++)
    {
        // ignore unfinished IMU factors
        if (it->to_idx_ < 0)
            continue;


        SALSA_ASSERT(it->from_idx_ == prev_idx, "Out-of-order IMU intervals, expected: %d, got %d, tail: %d, head: %d",
                     prev_idx, it->from_idx_, xbuf_tail_, xbuf_head_);
        SALSA_ASSERT(it->to_idx_ == (it->from_idx_ + 1) % STATE_BUF_SIZE, "Skipping States! from %d to %d ", it->from_idx_, it->to_idx_);
        prev_idx = it->to_idx_;
        SALSA_ASSERT(inWindow(it->to_idx_), "Trying to add IMU factor to node outside of window");
        SALSA_ASSERT(inWindow(it->from_idx_), "Trying to add IMU factor to node outside of window");
        SALSA_ASSERT(std::abs(1.0 - xbuf_[it->to_idx_].x.q().arr_.norm()) < 1e-8, "Quat Left Manifold");
        FunctorShield<ImuFunctor>* ptr = new FunctorShield<ImuFunctor>(&*it);
        problem.AddResidualBlock(new ImuFactorAD(ptr),
                                 NULL,
                                 xbuf_[it->from_idx_].x.data(),
                xbuf_[it->to_idx_].x.data(),
                xbuf_[it->from_idx_].v.data(),
                xbuf_[it->to_idx_].v.data(),
                imu_bias_.data());
    }
    SALSA_ASSERT(prev_idx == xbuf_head_, "not enough intervals");
}

void Salsa::addMocapFactors(ceres::Problem &problem)
{
    for (auto it = mocap_.begin(); it != mocap_.end(); it++)
    {
        SALSA_ASSERT(inWindow(it->idx_), "Trying to add Mocap factor to node outside of window");
        FunctorShield<MocapFunctor>* ptr = new FunctorShield<MocapFunctor>(&*it);
        problem.AddResidualBlock(new MocapFactorAD(ptr),
                                 NULL,
                                 xbuf_[it->idx_].x.data());
    }
}

void Salsa::addRawGnssFactors(ceres::Problem &problem)
{
    if (disable_gnss_)
        return;

    std::unordered_map<int, double*> prev_sw;
    for (auto pvec = prange_.begin(); pvec != prange_.end(); pvec++)
    {
        for (int i = 0; i < pvec->size(); i++)
        {
            PseudorangeFunctor &p((*pvec)[i]);
            if(!inWindow(p.idx_))
            {
                SD(5, "Trying to add GNSS factor to node outside of window. idx=%d, tail=%d, head=%d", p.idx_, xbuf_tail_, xbuf_head_);
                continue;
            }
            problem.AddResidualBlock(new PseudorangeFactor(&p),
                                     NULL,
                                     xbuf_[p.idx_].x.data(),
                                     xbuf_[p.idx_].v.data(),
                                     xbuf_[p.idx_].tau.data(),
                                     x_e2n_.data(),
                                     &(p.sw));

            // If we had a previous switching factor on this satellite,
            // then add the switching factor dynamics
            auto prev_sw_it = prev_sw.find(p.sat_id_);
            if (prev_sw_it != prev_sw.end())
            {
                problem.AddResidualBlock(new SwitchFactor(switchdot_Xi_),
                                         NULL,
                                         &(p.sw),
                                         prev_sw_it->second);
            }
            prev_sw[p.sat_id_] = &(p.sw);
        }

    }
    for (auto it = clk_.begin(); it != clk_.end(); it++)
    {
        if (it->to_idx_ < 0)
            continue;
        FunctorShield<ClockBiasFunctor>* ptr = new FunctorShield<ClockBiasFunctor>(&*it);
        problem.AddResidualBlock(new ClockBiasFactorAD(ptr),
                                 NULL,
                                 xbuf_[it->from_idx_].tau.data(),
                                 xbuf_[it->to_idx_].tau.data());
    }

}

void Salsa::addFeatFactors(ceres::Problem &problem)
{
    if (disable_vision_)
        return;

    FeatMap::iterator ft = xfeat_.begin();
    while (ft != xfeat_.end())
    {
        if (ft->second.funcs.size() < 2)
        {
            ft++;
            continue;
        }
        FeatDeque::iterator func = ft->second.funcs.begin();
        while (func != ft->second.funcs.end())
        {
            SALSA_ASSERT(inWindow(ft->second.idx0), "Trying to add factor to node outside of window: %d", ft->second.idx0);
            SALSA_ASSERT(inWindow(func->to_idx_), "Trying to add factor to node outside of window: %d", func->to_idx_);
            FeatFactor* ptr = new FeatFactor(&*func);
            problem.AddResidualBlock(ptr,
//                                     new ceres::CauchyLoss(1.0),
                                     NULL,
                                     xbuf_[ft->second.idx0].x.data(),
                    xbuf_[func->to_idx_].x.data(),
                    &ft->second.rho);
            func++;
        }
        ft++;
    }
}

void Salsa::addZeroVelFactors(ceres::Problem &problem)
{
    if (!enable_static_start_)
        return;

    int end = (xbuf_head_ + 1) % STATE_BUF_SIZE;
    for (int idx = xbuf_tail_; idx != end; idx = (idx + 1) % STATE_BUF_SIZE)
    {
        if (xbuf_[idx].t > static_start_end_)
            break;
        else
        {
            ZeroVelFunctor* ptr = new ZeroVelFunctor(x0_, v0_, zero_vel_Xi_);
            problem.AddResidualBlock(new ZeroVelFactorAD(ptr),
                                     NULL,
                                     xbuf_[idx].x.data(),
                                     xbuf_[idx].v.data());
        }
    }

}

void Salsa::initSolverOptions()
{
    options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.minimizer_progress_to_stdout = false ;
}

void Salsa::solve()
{
    ceres::Problem* problem = new ceres::Problem();

    addParameterBlocks(*problem);
    setAnchors(*problem);
    addImuFactors(*problem);
//    addFeatFactors(*problem);
//    addMocapFactors(*problem);
    addRawGnssFactors(*problem);
//    addZeroVelFactors(*problem);

    if (!disable_solver_)
        ceres::Solve(options_, problem, &summary_);
    //    std::cout << summary_.FullReport() << std::endl;

    delete problem;

    SD_S(3, "Finished Solve Iteration t: " << xhead().t << " p: [" << xhead().x.t().transpose()
             << "] att: [" << xhead().x.q().euler().transpose() << "]");

    logState();
    logOptimizedWindow();
    logRawGNSSRes();
    logFeatRes();
    logMocapRes();
    logFeatures();
    logSatPos();
    logPrangeRes();
}

}
