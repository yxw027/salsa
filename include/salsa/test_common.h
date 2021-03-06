#pragma once
#include <gtest/gtest.h>
#include <cmath>
#include <yaml-cpp/yaml.h>
#include <string>
#include <fstream>

#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)

#define ASSERT_MAT_EQ(v1, v2) \
{ \
    ASSERT_EQ((v1).rows(), (v2).rows()); \
    ASSERT_EQ((v1).cols(), (v2).cols()); \
    for (int row = 0; row < (v1).rows(); row++) {\
        for (int col = 0; col < (v2).cols(); col++) {\
            ASSERT_FLOAT_EQ((v1)(row, col), (v2)(row,col));\
        }\
    }\
}

#define ASSERT_MAT_NEAR(v1, v2, tol) \
{ \
    ASSERT_EQ((v1).rows(), (v2).rows()); \
    ASSERT_EQ((v1).cols(), (v2).cols()); \
    for (int row = 0; row < (v1).rows(); row++) {\
        for (int col = 0; col < (v2).cols(); col++) {\
            if (std::abs((v1)(row, col) - (v2)(row,col)) > (tol)) \
                cout << "[ " << row << ", " << col << " ] ";\
            ASSERT_NEAR((v1)(row, col), (v2)(row,col), (tol));\
        }\
    }\
}

#define EXPECT_MAT_NEAR(v1, v2, tol) \
{ \
    EXPECT_EQ((v1).rows(), (v2).rows()); \
    EXPECT_EQ((v1).cols(), (v2).cols()); \
    for (int row = 0; row < (v1).rows(); row++) {\
        for (int col = 0; col < (v2).cols(); col++) {\
            if (std::abs((v1)(row, col) - (v2)(row,col)) > tol) \
                std::cout << "[ " << row << ", " << col << " ] ";\
            EXPECT_NEAR((v1)(row, col), (v2)(row,col), (tol));\
        }\
    }\
}

#define ASSERT_XFORM_NEAR(x1, x2, tol) \
{ \
    ASSERT_NEAR((x1).t()(0), (x2).t()(0), tol);\
    ASSERT_NEAR((x1).t()(1), (x2).t()(1), tol);\
    ASSERT_NEAR((x1).t()(2), (x2).t()(2), tol);\
    ASSERT_NEAR((x1).q().w(), (x2).q().w(), tol);\
    ASSERT_NEAR((x1).q().x(), (x2).q().x(), tol);\
    ASSERT_NEAR((x1).q().y(), (x2).q().y(), tol);\
    ASSERT_NEAR((x1).q().z(), (x2).q().z(), tol);\
}

#define ASSERT_QUAT_NEAR(q1, q2, tol) \
do { \
    Vector3d qt = (q1) - (q2);\
    ASSERT_LE(std::abs(qt(0)), tol);\
    ASSERT_LE(std::abs(qt(1)), tol);\
    ASSERT_LE(std::abs(qt(2)), tol);\
} while(0)

#define EXPECT_QUAT_NEAR(q1, q2, tol) \
do { \
    Vector3d qt = (q1) - (q2);\
    EXPECT_LE(std::abs(qt(0)), tol);\
    EXPECT_LE(std::abs(qt(1)), tol);\
    EXPECT_LE(std::abs(qt(2)), tol);\
} while(0)


#define EXPECT_MAT_FINITE(mat) \
do {\
  for (int c = 0; c < (mat).cols(); c++) \
  { \
    for (int r = 0; r < (mat).rows(); r++) \
    { \
      EXPECT_TRUE(std::isfinite((mat)(r,c))); \
    } \
  }\
} while(0)

#define EXPECT_MAT_NAN(mat) \
do {\
  for (int c = 0; c < (mat).cols(); c++) \
    { \
      for (int r = 0; r < (mat).rows(); r++) \
      { \
        EXPECT_TRUE(std::isnan((mat)(r,c))); \
      } \
  }\
} while(0)


#define EXPECT_FINITE(val) EXPECT_TRUE(std::isfinite(val))


inline std::string default_params(const std::string& prefix, std::string label="test")
{
    std::string filename = "/tmp/Salsa.default.yaml";
    std::ofstream tmp(filename);
    YAML::Node node = YAML::LoadFile(SALSA_DIR"/params/salsa.yaml");
    node["x_b2m"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    node["x_b2o"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    node["x_b2c"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    node["tm"] = 0.0;
    node["tc"] = 0.0;
    node["log_prefix"] = prefix;
    node["enable_out_of_order"] = false;
    node["label"] = label;
    node["simulate_klt"] = false;
    node["enable_static_start"] = false;
    node["max_kf_window"] = 10;
    tmp << node;
    tmp.close();
    return filename;
}

inline std::string small_feat_test(const std::string& prefix, bool init_depth=true)
{
    std::string filename = "/tmp/Salsa.smallfeat.yaml";
    std::ofstream tmp(filename);
    YAML::Node node = YAML::LoadFile(SALSA_DIR"/params/salsa.yaml");
    node["x_b2m"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    node["x_b2c"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    node["x_b2o"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    node["tm"] = 0.0;
    node["tc"] = 0.0;
    node["log_prefix"] = prefix;
    node["N"] = 4;
    node["kf_feature_thresh"] = 0.80;
    node["kf_parallax_thresh"] = 500;
    node["num_feat"] = 4;
    node["use_measured_depth"] = init_depth;
    node["enable_out_of_order"] = false;
    node["enable_static_start"] = false;
    tmp << node;
    tmp.close();
    return filename;
}

inline std::string imu_only()
{
    std::string filename = "/tmp/Salsa.imu_only.yaml";
    std::ofstream tmp(filename);
    YAML::Node node = YAML::LoadFile(SALSA_DIR"/params/sim_params.yaml");
    node["imu_enabled"] =  true;
    node["alt_enabled"] =  false;
    node["baro_enabled"] =  false;
    node["mocap_enabled"] =  false;
    node["vo_enabled"] =  false;
    node["camera_enabled"] =  false;
    node["gnss_enabled"] =  false;
    node["raw_gnss_enabled"] =  false;
    node["enable_out_of_order"] = false;

    tmp << node;
    tmp.close();
    return filename;
}

inline std::string imu_mocap(bool noise=true)
{
  std::string filename = "/tmp/Salsa.imu_only.yaml";
  std::ofstream tmp(filename);
  YAML::Node node = YAML::LoadFile(SALSA_DIR"/params/sim_params.yaml");
  node["imu_enabled"] =  true;
  node["alt_enabled"] =  false;
  node["baro_enabled"] =  false;
  node["mocap_enabled"] =  true;
  node["vo_enabled"] =  false;
  node["camera_enabled"] =  false;
  node["gnss_enabled"] =  false;
  node["raw_gnss_enabled"] =  false;
  node["enable_out_of_order"] = false;

  if (!noise)
  {
      node["use_accel_truth"] = !noise;
      node["use_mocap_truth"] = !noise;
      node["use_gyro_truth"] = !noise;
      node["use_camera_truth"] = !noise;
      node["use_depth_truth"] = !noise;
      node["use_gnss_truth"] = !noise;
      node["use_raw_gnss_truth"] = !noise;
  }


  tmp << node;
  tmp.close();
  return filename;
}

inline std::string imu_feat(bool noise=true, double tmax=-1.0)
{
    std::string filename = "/tmp/Salsa.imu_only.yaml";
    std::ofstream tmp(filename);
    YAML::Node node = YAML::LoadFile(SALSA_DIR"/params/sim_params.yaml");
    if (tmax > 0)
        node["tmax"] = tmax;
    node["imu_enabled"] =  true;
    node["alt_enabled"] =  false;
    node["baro_enabled"] =  false;
    node["mocap_enabled"] =  false;
    node["vo_enabled"] =  false;
    node["camera_enabled"] =  true;
    node["gnss_enabled"] =  false;
    node["raw_gnss_enabled"] =  false;
    node["enable_out_of_order"] = false;

    if (!noise)
    {
        node["use_accel_truth"] = !noise;
        node["use_gyro_truth"] = !noise;
        node["use_camera_truth"] = !noise;
        node["use_depth_truth"] = !noise;
        node["use_gnss_truth"] = !noise;
        node["use_raw_gnss_truth"] = !noise;
    }

    tmp << node;
    tmp.close();
    return filename;
}

inline std::string imu_raw_gnss(bool noise=true, double tmax=-1.0)
{
    std::string filename = "/tmp/Salsa.imu_raw_gnss.yaml";
    std::ofstream tmp(filename);
    YAML::Node node = YAML::LoadFile(SALSA_DIR"/params/sim_params.yaml");
    if (tmax > 0)
        node["tmax"] = tmax;
    node["imu_enabled"] =  true;
    node["alt_enabled"] =  false;
    node["baro_enabled"] =  false;
    node["mocap_enabled"] =  false;
    node["vo_enabled"] =  false;
    node["camera_enabled"] =  false;
    node["gnss_enabled"] =  false;
    node["raw_gnss_enabled"] =  true;
    node["ephemeris_filename"] = SALSA_DIR"/sample/eph.dat";
    node["enable_out_of_order"] = false;

    if (!noise)
    {
        node["use_accel_truth"] = !noise;
        node["use_gyro_truth"] = !noise;
        node["use_camera_truth"] = !noise;
        node["use_depth_truth"] = !noise;
        node["use_gnss_truth"] = !noise;
        node["use_raw_gnss_truth"] = !noise;
    }

    tmp << node;
    tmp.close();
    return filename;
}

inline std::string imu_feat_gnss(bool noise=true, double tmax=-1.0)
{
    std::string filename = "/tmp/Salsa.imu_only.yaml";
    std::ofstream tmp(filename);
    YAML::Node node = YAML::LoadFile(SALSA_DIR"/params/sim_params.yaml");
    if (tmax > 0)
        node["tmax"] = tmax;
    node["imu_enabled"] =  true;
    node["alt_enabled"] =  false;
    node["baro_enabled"] =  false;
    node["mocap_enabled"] =  false;
    node["vo_enabled"] =  false;
    node["camera_enabled"] =  true;
    node["gnss_enabled"] =  false;
    node["raw_gnss_enabled"] =  true;
    node["ephemeris_filename"] = SALSA_DIR"/sample/eph.dat";
    node["enable_out_of_order"] = false;

    if (!noise)
    {
        node["use_accel_truth"] = !noise;
        node["use_gyro_truth"] = !noise;
        node["use_camera_truth"] = !noise;
        node["use_depth_truth"] = !noise;
        node["use_gnss_truth"] = !noise;
        node["use_raw_gnss_truth"] = !noise;
    }

    tmp << node;
    tmp.close();
    return filename;
}
