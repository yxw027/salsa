#pragma once

#include <memory>
#include <deque>
#include <functional>
#include <experimental/filesystem>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "geometry/xform.h"
#include "multirotor_sim/estimator_base.h"
#include "multirotor_sim/utils.h"
#include "gnss_utils/satellite.h"
#include "gnss_utils/wgs84.h"

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

using multirotor_sim::VecMat3;
using multirotor_sim::VecVec3;
using multirotor_sim::SatVec;
using multirotor_sim::ImageFeat;

namespace salsa
{

class MTLogger;
class Logger;
class Salsa : public multirotor_sim::EstimatorBase
{
public:
    typedef Eigen::Matrix<double, 11, 1> Vector11d;
    typedef std::deque<MocapFunctor, Eigen::aligned_allocator<MocapFunctor>> MocapDeque;
    typedef std::vector<PseudorangeFunctor, Eigen::aligned_allocator<PseudorangeFunctor>> PseudorangeVec;
    typedef std::deque<PseudorangeVec, Eigen::aligned_allocator<PseudorangeVec>> PseudorangeDeque;
    typedef std::deque<ImuFunctor, Eigen::aligned_allocator<ImuFunctor>> ImuDeque;
    typedef std::deque<ClockBiasFunctor, Eigen::aligned_allocator<ClockBiasFunctor>> ClockBiasDeque;
    typedef std::vector<gnss_utils::Satellite, Eigen::aligned_allocator<gnss_utils::Satellite>> SatVec;
    typedef std::map<int,Feat,std::less<int>,Eigen::aligned_allocator<std::pair<const int,Feat>>> FeatMap;


    Salsa();
    ~Salsa();

    void init(const std::string& filename);

    void load(const std::string& filename);
    void initImg(const std::string& filename);
    void initState();
    void initFactors();
    void initialize(const double& t, const xform::Xformd &x0, const Eigen::Vector3d& v0, const Eigen::Vector2d& tau0);
    void initSolverOptions();
    void setInitialState(const xform::Xformd& x0);

    void initLog(const std::string &filename);
    void logRawGNSSRes();
    void logFeatRes();
    void logMocapRes();
    void logFeatures();
    void logOptimizedWindow();
    void logState();
    void logSatPos();
    void logPrangeRes();
    void logCurrentState();
    void logImu();
    void logXe2n();

    void endInterval(double t);
    void startNewInterval(double t);
    void cleanUpSlidingWindow();
    void setNewKeyframe();
    const State& lastKfState();
    bool calcNewKeyframeCondition(const Features& z);
    void cleanUpFeatureTracking();
    void rmLostFeatFromKf();

    void solve();
    void addParameterBlocks(ceres::Problem& problem);
    void addImuFactors(ceres::Problem& problem);
    void addMocapFactors(ceres::Problem& problem);
    void setAnchors(ceres::Problem& problem);
    void addRawGnssFactors(ceres::Problem& problem);
    void addFeatFactors(ceres::Problem& problem);



    void imuCallback(const double &t, const Vector6d &z, const Matrix6d &R) override;
    void mocapCallback(const double &t, const xform::Xformd &z, const Matrix6d &R) override;
    void rawGnssCallback(const gnss_utils::GTime& t, const VecVec3& z, const VecMat3& R,
                         SatVec &sat, const std::vector<bool>& slip) override;
    void imageCallback(const double& t, const ImageFeat& z, const Eigen::Matrix2d& R_pix,
                       const Matrix1d& R_depth) override;
    void imageCallback(const double& t, const Features& z, const Eigen::Matrix2d& R_pix, bool new_keyframe);

    void imageCallback(const double &t, const cv::Mat& img, const Eigen::Matrix2d &R_pix);
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

    void initGNSS(const std::string& filename);
    int getSatIdx(int sat_id) const;
    void ephCallback(const gnss_utils::GTime &t, const gnss_utils::eph_t& eph);
    void refreshSatIdx();
    void obsCallback(const ObsVec& obs);
    void filterObs(const ObsVec& obs);
    void pointPositioning(const gnss_utils::GTime& t, const ObsVec &obs, SatVec &sat, Vector8d &xhat) const;

    enum {
        NOT_NEW_KEYFRAME = 0,
        FIRST_KEYFRAME = 1,
        TOO_MUCH_PARALLAX = 2,
        INSUFFICIENT_MATCHES = 3
    };
    int kf_condition_;
    std::function<void(int kf, int condition)> new_kf_cb_ = nullptr;

    State current_state_;
    xform::Xformd x0_;

    int xbuf_head_, xbuf_tail_;

    bool disable_solver_;

    StateVec xbuf_;
    std::vector<double> s_; int ns_;
    Vector6d imu_bias_;
    int current_node_;
    int oldest_node_;
    int current_kf_;



    int STATE_BUF_SIZE;

    double switch_weight_;
    double acc_wander_weight_;
    double gyro_wander_weight_;
    Matrix6d acc_bias_xi_;
    State::dxMat state_anchor_xi_;
    Matrix6d x_e2n_anchor_xi_;
    Matrix6d x_u2c_anchor_xi_;

    ImuDeque imu_;
    ImuBiasAnchor* bias_;
    StateAnchor* state_anchor_;
    XformAnchor* x_e2n_anchor_;
    XformAnchor* x_u2c_anchor_;
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

    struct log
    {
        enum
        {
            CurrentState,
            Opt,
            RawRes,
            FeatRes,
            Feat,
            State,
            CB,
            MocapRes,
            SatPos,
            PRangeRes,
            Imu,
            Xe2n,
            NumLogs,
        };
    };
    std::vector<Logger*> logs_;

    std::string log_prefix_;
    Camera<double> cam_;
    xform::Xformd x_u2m_; // transform from imu to mocap frame
    xform::Xformd x_u2b_; // transform from imu to body frame
    xform::Xformd x_u2c_; // transform from imu to camera frame
    xform::Xformd x_e2n_; // transform from ECEF to NED (inertial) frame
    double dt_m_; // time offset of mocap  (t_m(stamped) - dt_m = t(true))
    double dt_c_; // time offset of camera (t_c(stamped) - dt_c = t(true))
    gnss_utils::GTime start_time_;
    Eigen::Matrix2d clk_bias_Xi_;
    bool disable_mocap_;


    bool disable_gnss_;
    bool estimate_origin_;
    double doppler_cov_;
    bool use_point_positioning_;
    double min_satellite_elevation_;
    double kf_parallax_thresh_;
    double kf_feature_thresh_;
    Features kf_feat_, current_feat_;
    int kf_Nmatch_feat_ = -1;
    double kf_parallax_;
    int kf_num_feat_;
    bool use_measured_depth_;

    int node_window_;

    bool sim_KLT_;

    // KLT Tracker
    int got_first_img_;
    bool show_matches_;
    int feature_nearby_radius_;
    int next_feature_id_;
    std::vector<uchar> match_status_;
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
//typedef struct
//{
//    gnss_utils::GTime g;
//    double range; // pseudorange
//    double rate;
//    double d; // geometric distance
//    double azel[2];
//    double iono_delay;
//} range_t;
//typedef struct
//{
//    int enable;
//    int vflg;
//    double alpha0,alpha1,alpha2,alpha3;
//    double beta0,beta1,beta2,beta3;
//} ionoutc_t;
//void eph2pos(const gnss_utils::GTime& t, const gnss_utils::eph_t *eph, Eigen::Vector3d& pos, double *dts);
//double ionmodel(const gnss_utils::GTime& t, const double *pos, const double *azel);
//double ionosphericDelay(const ionoutc_t *ionoutc, GTime g, double *llh, double *azel);
//void computeRange(range_t *rho, gnss_utils::Satellite &eph, ionoutc_t *ionoutc, gnss_utils::GTime g, Vector3d& xyz);
