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
#include "factors/switch_dynamics.h"
#include "factors/carrier_phase.h"
#include "factors/clock_dynamics.h"
#include "factors/anchor.h"
#include "factors/feat.h"

#include "salsa/logger.h"
#include "salsa/state.h"
#include "salsa/feat.h"
#include "salsa/meas.h"

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

    /************************************/
    /*          Initialization          */
    /************************************/
    void init(const std::string& filename);
    void load(const std::string& filename);
    void initImg(const std::string& filename);
    void initState();
    void initFactors();
    void initialize(const double& t, const xform::Xformd &x0, const Eigen::Vector3d& v0, const Eigen::Vector2d& tau0);
    void initSolverOptions();
    void setInitialState(const xform::Xformd& x0);
    // Constants
    xform::Xformd x0_;
    Eigen::Vector3d v0_;
    Camera<double> cam_;
    xform::Xformd x_b2m_; // transform from imu to mocap frame
    xform::Xformd x_b2c_; // transform from imu to camera frame
    double dt_m_; // time offset of mocap  (t_m(stamped) - dt_m = t(true))
    double dt_c_; // time offset of camera (t_c(stamped) - dt_c = t(true))
    Eigen::Vector3d p_b2g_; // Position of gps antenna wrt body
    xform::Xformd x_b2o_; // transform from body frame to output
    gnss_utils::GTime start_time_;

    /************************************/
    /*             Logging              */
    /************************************/
    void initLog(const std::string &filename);
    void logRawGNSSRes();
    void logFeatRes();
    void logMocapRes();
    void logFeatures();
    void logOptimizedWindow();
    void logState();
    void logSwParams();
    void logSatPos();
    void logPrangeRes();
    void logCurrentState();
    void logImu();
    void printGraph();
    void printFeat();
    void printImuIntervals();
    struct log
    {
        enum
        {
            CurrentState,
            Opt,
            SwParams,
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
            Graph,
            NumLogs,
        };
    };
    std::vector<Logger*> logs_;
    std::string log_prefix_;


    /************************************/
    /*         Graph Management         */
    /************************************/
//    void endInterval(double t);
    void startNewInterval(double t);
    void cleanUpSlidingWindow();
    void setNewKeyframe();
    const State& lastKfState();
    bool calcNewKeyframeCondition(const Features& z);
    void cleanUpFeatureTracking(int oldest_kf_idx);
    void rmLostFeatFromKf();
    bool inWindow(int idx);
    State& xhead() { return xbuf_[xbuf_head_]; }
    ImuIntegrator current_state_integrator_;
    State current_state_;
    int current_node_;
    int oldest_node_;
    int oldest_kf_;
    int current_kf_;
    int STATE_BUF_SIZE;
    int max_node_window_;
    int max_kf_window_;

    // Estimated variables
    int xbuf_head_, xbuf_tail_;
    StateVec xbuf_;
    ClockBiasDeque clk_;
    xform::Xformd x_e2n_; // transform from ECEF to inertial frame
    Vector6d imu_bias_;

    // Factors
    ImuDeque imu_;
    ImuBiasAnchor* bias_;
    FeatMap xfeat_; int nf_;
    StateAnchor* state_anchor_;
    State::dxMat state_anchor_xi_;
    XformAnchor* x_e2n_anchor_;
    Matrix6d x_e2n_anchor_xi_;
    Eigen::Matrix2d clk_bias_Xi_;
    double switch_Xi_;
    double switchdot_Xi_;
    Matrix6d imu_bias_xi_;
    MocapDeque mocap_;
    PseudorangeDeque prange_;


    /************************************/
    /*           Optimization           */
    /************************************/
    void solve();
    void addParameterBlocks(ceres::Problem& problem);
    void addImuFactors(ceres::Problem& problem);
    void addMocapFactors(ceres::Problem& problem);
    void setAnchors(ceres::Problem& problem);
    void addRawGnssFactors(ceres::Problem& problem);
    void addFeatFactors(ceres::Problem& problem);
    bool disable_solver_;
    bool disable_mocap_;
    bool disable_gnss_;
    bool disable_vision_;
    ceres::Solver::Options options_;
    ceres::Solver::Summary summary_;

    /************************************/
    /*               IMU                */
    /************************************/
    void imuCallback(const double &t, const Vector6d &z, const Matrix6d &R) override;
    bool checkIMUOrder();
    double acc_wander_weight_;
    double gyro_wander_weight_;


    /************************************/
    /*               Mocap              */
    /************************************/
    void mocapCallback(const double &t, const xform::Xformd &z, const Matrix6d &R) override;
    void mocapUpdate(const meas::Mocap& m);
    void initializeStateMocap(const meas::Mocap& m);
    bool update_on_mocap_;

    /************************************/
    /*               GNSS               */
    /************************************/
    void initGNSS(const std::string& filename);
    int getSatIdx(int sat_id) const;
    void ephCallback(const gnss_utils::GTime &t, const gnss_utils::eph_t& eph);
    void refreshSatIdx();
    void obsCallback(const ObsVec& obs);
    void filterObs(const ObsVec& obs);
    void pointPositioning(const gnss_utils::GTime& t, const ObsVec &obs, SatVec &sat, Vector8d &xhat) const;
    void rawGnssCallback(const gnss_utils::GTime& t, const VecVec3& z, const VecMat3& R,
                         SatVec &sat, const std::vector<bool>& slip) override;
    void gnssUpdate(const meas::Gnss& m);
    void initializeStateGnss(const meas::Gnss& m);
    int ns_;
    bool update_on_gnss_;
    double doppler_cov_;
    bool use_point_positioning_;
    double min_satellite_elevation_;
    SatVec sats_;
    ObsVec filtered_obs_;
    int n_obs_;
    bool enable_switching_factors_;

    /************************************/
    /*              Image               */
    /************************************/
    void imageCallback(const double &tc, const cv::Mat& img, const Eigen::Matrix2d &R_pix);
    void imageCallback(const double& tc, const ImageFeat& z, const Eigen::Matrix2d& R_pix,
                       const Matrix1d& R_depth) override; // bindings for simulator image
    void imageUpdate(const meas::Img& m);
    int numTotalFeat() const;

    enum {
        NOT_NEW_KEYFRAME = 0,
        FIRST_KEYFRAME = 1,
        TOO_MUCH_PARALLAX = 2,
        INSUFFICIENT_MATCHES = 3
    };
    double kf_parallax_thresh_;
    double kf_feature_thresh_;
    Features kf_feat_, current_feat_;
    int kf_Nmatch_feat_ = -1;
    double kf_parallax_;
    int kf_num_feat_;
    bool use_measured_depth_;
    int kf_condition_;
    bool update_on_camera_;
    std::function<void(int kf, int condition)> new_kf_cb_ = nullptr;

    /************************************/
    /*               KLT                */
    /************************************/
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
    bool sim_KLT_;
    double t_next_klt_output_;
    double tracker_freq_;
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

    /************************************/
    /*            Meas Buffer           */
    /************************************/
    void handleMeas();
    void integrateTransition(double t);
    void initializeNodeWithImu();
    void initializeNodeWithGnss(const meas::Gnss& m);
    void initializeNodeWithMocap(const meas::Mocap& mocap);
    void initialize(const meas::Base* m);
    void addMeas(const meas::Mocap&& mocap);
    void addMeas(const meas::Gnss&& gnss);
    void addMeas(const meas::Img&& img);
    std::multiset<meas::Base*, std::function<bool(const meas::Base*, const meas::Base*)>> new_meas_;
    std::deque<meas::Imu, Eigen::aligned_allocator<meas::Imu>> imu_meas_buf_;
    std::deque<meas::Mocap, Eigen::aligned_allocator<meas::Mocap>> mocap_meas_buf_;
    std::deque<meas::Gnss, Eigen::aligned_allocator<meas::Gnss>> gnss_meas_buf_;
    std::deque<meas::Img, Eigen::aligned_allocator<meas::Img>> img_meas_buf_;

    /**************************************/
    /*            Meas Buffer 2           */
    /**************************************/
    void handleMeas2();
    void alignIMU();
    void findInterval(double t);
    void splitInterval(int interval_idx, double t);
    void moveNode(double t);
    int newNode(double t);
};
}
