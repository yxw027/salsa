#pragma once

#include <cmath>
#include <stdio.h>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <string>
#include <fstream>

//#ifndef NDEBUG
#define SALSA_ASSERT(condition, ...) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": "; \
            fprintf(stderr, __VA_ARGS__);\
            std::cerr << std::endl; \
            assert(condition); \
        } \
    } while (false)
//#else
//#   define SALSA_ASSERT(...)
//#endif

#ifndef DEBUGPRINT
#define DEBUGPRINT 1
#endif

#define DEBUGLEVEL 3

#if DEBUGPRINT
#define SL std::cout << __LINE__ << std::endl
#define SD(level, f_, ...) do{ \
    if ((level) > DEBUGLEVEL) {\
        printf((f_), ##__VA_ARGS__);\
        printf("\n");\
    }\
} while(false)

#define SD_S(args) std::cout << args;
#else
#define SL
#define SD(...)
#define SD_S(...)
#endif

inline std::string default_params(const std::string& prefix)
{
    std::string filename = "/tmp/Salsa.default.yaml";
    std::ofstream tmp(filename);
    YAML::Node node = YAML::LoadFile(SALSA_DIR"/params/salsa.yaml");
    node["X_u2m"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    node["X_u2c"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    node["q_u2b"] = std::vector<double>{1, 0, 0, 0};
    node["tm"] = 0.0;
    node["tc"] = 0.0;
    node["log_prefix"] = prefix;
    tmp << node;
    tmp.close();
    return filename;
}

inline std::string small_feat_test(const std::string& prefix, bool init_depth=true)
{
    std::string filename = "/tmp/Salsa.smallfeat.yaml";
    std::ofstream tmp(filename);
    YAML::Node node = YAML::LoadFile(SALSA_DIR"/params/salsa.yaml");
    node["X_u2m"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    node["X_u2c"] = std::vector<double>{0, 0, 0, 1, 0, 0, 0};
    node["q_u2b"] = std::vector<double>{1, 0, 0, 0};
    node["tm"] = 0.0;
    node["tc"] = 0.0;
    node["log_prefix"] = prefix;
    node["N"] = 4;
    node["kf_feature_thresh"] = 0.80;
    node["kf_parallax_thresh"] = 500;
    node["num_feat"] = 4;
    node["use_measured_depth"] = init_depth;
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

    tmp << node;
    tmp.close();
    return filename;
}

inline std::string imu_mocap()
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

  tmp << node;
  tmp.close();
  return filename;
}

inline std::string imu_feat(bool noise=false, double tmax=-1.0)
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

    node["use_camera_truth"] = !noise;
    node["use_depth_truth"] = !noise;

    tmp << node;
    tmp.close();
    return filename;
}

inline std::string imu_raw_gnss()
{
    std::string filename = "/tmp/Salsa.imu_raw_gnss.yaml";
    std::ofstream tmp(filename);
    YAML::Node node = YAML::LoadFile(SALSA_DIR"/params/sim_params.yaml");
    node["imu_enabled"] =  true;
    node["alt_enabled"] =  false;
    node["baro_enabled"] =  false;
    node["mocap_enabled"] =  false;
    node["vo_enabled"] =  false;
    node["camera_enabled"] =  false;
    node["gnss_enabled"] =  false;
    node["raw_gnss_enabled"] =  true;
    node["ephemeris_filename"] = SALSA_DIR"/sample/eph.dat";

    tmp << node;
    tmp.close();
    return filename;
}
