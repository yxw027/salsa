#pragma once

#include <memory>
#include <deque>
#include <functional>
#include <experimental/filesystem>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "geometry/xform.h"
#include "multirotor_sim/estimator_base.h"
#include "multirotor_sim/satellite.h"
#include "multirotor_sim/wsg84.h"
#include "multirotor_sim/utils.h"

#include "factors/imu.h"
#include "factors/mocap.h"
#include "factors/xform.h"
#include "factors/pseudorange.h"
#include "factors/clock_dynamics.h"
#include "factors/carrier_phase.h"
#include "factors/clock_dynamics.h"
#include "factors/anchor.h"
#include "factors/feat.h"

#include "salsa/logger.h"
#include "salsa/state.h"

#include "opencv2/opencv.hpp"

using namespace std;
using namespace Eigen;
using namespace xform;
using multirotor_sim::VecMat3;
using multirotor_sim::VecVec3;
using multirotor_sim::ImageFeat;

namespace salsa
{

class MTLogger;
class Logger;
class Salsa : public multirotor_sim::EstimatorBase
{
public:
    typedef Matrix<double, 11, 1> Vector11d;
    typedef std::deque<MocapFunctor, aligned_allocator<MocapFunctor>> MocapDeque;
    typedef std::vector<PseudorangeFunctor, aligned_allocator<PseudorangeFunctor>> PseudorangeVec;
    typedef std::deque<PseudorangeVec, aligned_allocator<PseudorangeVec>> PseudorangeDeque;
    typedef std::deque<ImuFunctor, aligned_allocator<ImuFunctor>> ImuDeque;
    typedef std::deque<ClockBiasFunctor, aligned_allocator<ClockBiasFunctor>> ClockBiasDeque;
    typedef std::vector<Satellite, aligned_allocator<Satellite>> SatVec;
    typedef std::map<int,Feat,std::less<int>,aligned_allocator<std::pair<const int,Feat>>> FeatMap;


    Salsa();
    ~Salsa();

    void init(const std::string& filename);

    void load(const std::string& filename);
    void loadKLT(const std::string& filename);
    void initState();
    void initFactors();
    void initialize(const double& t, const Xformd &x0, const Vector3d& v0, const Vector2d& tau0);
    void initSolverOptions();

    void initLog();
    void logRawGNSSRes();
    void logFeatRes();
    void logMocapRes();
    void logFeatures();
    void logOptimizedWindow();
    void logState();
    void logCurrentState();
    void renderGraph(const std::string& filename);

    void finishNode(const double& t, bool new_node, bool new_keyframe);
    void cleanUpSlidingWindow();
    const State& lastKfState();
    bool calcNewKeyframeCondition(const Features& z);
    void cleanUpFeatureTracking(int new_from_idx, int oldest_desired_kf);
    void rmLostFeatFromKf();

    void solve();
    void addParameterBlocks(ceres::Problem& problem);
    void addImuFactors(ceres::Problem& problem);
    void addMocapFactors(ceres::Problem& problem);
    void addOriginConstraint(ceres::Problem& problem);
    void addRawGnssFactors(ceres::Problem& problem);
    void addFeatFactors(ceres::Problem& problem);


    void pointPositioning(const GTime& t, const VecVec3& z,
                          std::vector<Satellite>& sat, Vector8d &xhat) const;

    void imuCallback(const double &t, const Vector6d &z, const Matrix6d &R) override;
    void mocapCallback(const double &t, const Xformd &z, const Matrix6d &R) override;
    void rawGnssCallback(const GTime& t, const VecVec3& z, const VecMat3& R,
                         std::vector<Satellite>& sat, const std::vector<bool>& slip) override;
    void imageCallback(const double& t, const ImageFeat& z, const Matrix2d& R_pix,
                       const Matrix1d& R_depth) override;
    void imageCallback(const double& t, const Features& z, const Matrix2d& R_pix, bool new_keyframe);

    void imageCallback(const double &t, const cv::Mat& img, const Matrix2d &R_pix);
    bool dropFeature(int idx);
    void setFeatureMask(const std::string& filename);
    void showImage();
    void collectNewfeatures();
    void trackFeatures();
    void filterFeaturesRANSAC();
    void filterFeaturesOutOfBounds();
    void filterFeaturesTooClose();
    void createNewKeyframe();
    int calcNewKeyframeConditionKLT();
    void calcCurrentZetas();
    bool isTrackedFeature(int id) const;

    int getSatIdx(int sat_id) const;
    void ephCallback(const eph_t& eph);
    void filterObs(const ObsVec& obs);
    void rawGnssCallback();

    enum {
        NOT_NEW_KEYFRAME = 0,
        FIRST_KEYFRAME = 1,
        TOO_MUCH_PARALLAX = 2,
        INSUFFICIENT_MATCHES = 3
    };
    int kf_condition_;
    std::function<void(int kf, int condition)> new_kf_cb_ = nullptr;

    State current_state_;

    int xbuf_head_, xbuf_tail_;

    bool disable_solver_;

    StateVec xbuf_;
    vector<double> s_; int ns_;
    Vector6d imu_bias_;
    int current_node_;
    int oldest_node_;
    int current_kf_;


    int STATE_BUF_SIZE;

    double switch_weight_;
    double acc_wander_weight_;
    double gyro_wander_weight_;
    Matrix6d acc_bias_xi_;
    Matrix11d anchor_xi_;

    ImuDeque imu_;
    ImuBiasDynamicsFunctor* bias_;
    AnchorFunctor* anchor_;
    MocapDeque mocap_;
    PseudorangeDeque prange_;
    ClockBiasDeque clk_;
    FeatMap xfeat_; int nf_;

    ceres::Solver::Options options_;
    ceres::Solver::Summary summary_;

    enum{
        NONE,
        IMG,
        GNSS,
        MOCAP
    };
    int last_callback_;
    Logger* current_state_log_ = nullptr;
    Logger* opt_log_ = nullptr;
    Logger* raw_gnss_res_log_ = nullptr;
    Logger* feat_res_log_ = nullptr;
    Logger* feat_log_ = nullptr;
    Logger* state_log_ = nullptr;
    Logger* cb_log_ = nullptr;
    Logger* mocap_res_log_ = nullptr;

    std::string log_prefix_;
    Camera<double> cam_;
    Xformd x_u2m_; // transform from imu to mocap frame
    Xformd x_u2b_; // transform from imu to body frame
    Xformd x_u2c_; // transform from imu to camera frame
    Xformd x_e2n_; // transform from ECEF to NED (inertial) frame
    double dt_m_; // time offset of mocap  (t(stamped) - dt_m = t(true))
    double dt_c_; // time offset of camera (t(stamped) - dt_m = t(true))
    GTime start_time_;
    Matrix2d clk_bias_Xi_;

    double kf_parallax_thresh_;
    double kf_feature_thresh_;
    Features kf_feat_, current_feat_;
    int kf_Nmatch_feat_;
    double kf_parallax_;
    int kf_num_feat_;
    bool use_measured_depth_;

    int N_;

    // KLT Tracker
    int got_first_img_;
    bool show_matches_;
    int feature_nearby_radius_;
    int next_feature_id_;
    vector<uchar> match_status_;
    std::vector<cv::Point2f> prev_features_;
    std::vector<cv::Scalar> colors_;
    std::vector<uchar> status_;
    cv::Mat kf_img_;
    cv::Mat prev_img_;
    cv::Mat current_img_;
    cv::Mat color_img_;
    cv::Mat mask_;
    cv::Mat point_mask_;

    // Satellite Manager
    SatVec sats_;
    ObsVec filtered_obs_;
    int n_obs_;


};
}
