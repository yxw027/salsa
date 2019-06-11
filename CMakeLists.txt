cmake_minimum_required(VERSION 2.8.3)
project(salsa)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nanoflann_eigen REQUIRED)
find_package(lin_alg_tools REQUIRED)
find_package(geometry REQUIRED)
find_package(gnss_utils REQUIRED)
find_package(multirotor_sim REQUIRED)

add_definitions(-DSALSA_DIR="${CMAKE_CURRENT_LIST_DIR}")

add_library(factors
    src/factors/xform.cpp
    src/factors/imu.cpp
    src/factors/mocap.cpp
    src/factors/feat.cpp
    src/factors/anchor.cpp
    src/factors/clock_dynamics.cpp
    src/factors/switch_dynamics.cpp
    src/factors/pseudorange.cpp)
target_link_libraries(factors
    ${geometry_LIBRARIES}
    ${gnss_utils_LIBRARIES}
    ${CERES_LIBRARIES})
target_include_directories(factors PUBLIC
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS})

add_library(salsa
    src/state.cpp
    src/meas.cpp
    src/salsa.cpp
    src/salsa_mocap.cpp
    src/salsa_gnss.cpp
    src/salsa_vision.cpp
    src/salsa_log.cpp
    src/klt.cpp
    src/feat.cpp
    src/num_diff.cpp)
target_link_libraries(salsa
    factors
    stdc++fs
    ${geometry_LIBRARIES}
    ${gnss_utils_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES})


option (BUILD_ROS "Build ROS Tools" ON)
if (BUILD_ROS)
    find_package(catkin REQUIRED COMPONENTS
        rosbag
        roscpp
        sensor_msgs
        inertial_sense
        nav_msgs)
    message(STATUS "Building salsa ROS library")
    add_library(salsa_ros src/salsa_ros.cpp)
    target_include_directories(salsa_ros PUBLIC ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
    target_link_libraries(salsa_ros salsa ${catkin_LIBRARIES})

    add_executable(salsa_rosbag src/salsa_rosbag.cpp)
    target_include_directories(salsa_rosbag PUBLIC ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
    target_link_libraries(salsa_rosbag salsa_ros ${catkin_LIBRARIES})

#    add_executable(salsa_node src/salsa_node.cpp)
#    target_link_libraries(salsa_node salsa_ros ${catkin_LIBRARIES})
#    target_include_directories(salsa_node PUBLIC ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
endif()


option (BUILD_TESTS "Build Tests" ON)
if (BUILD_TESTS)
    find_package(GTest REQUIRED)
    message(STATUS "Building salsa tests")
    add_executable(test_salsa
        src/test/test_salsa.cpp
        src/test/test_mocap_factor.cpp
        src/test/test_imu_factor.cpp
        src/test/test_logger.cpp
        src/test/test_point_positioning.cpp
        src/test/test_feat_factor.cpp
        src/test/test_misc.cpp
        src/test/test_vision.cpp
        src/test/test_klt.cpp
        src/test/test_xform_local_param.cpp
        src/test/test_pseudorange_factor.cpp
     )
     target_link_libraries(test_salsa
        salsa
        gtest
        gtest_main
        ${geometry_LIBRARIES}
        ${gnss_utils_LIBRARIES}
        ${multirotor_sim_LIBRARIES}
        ${GTEST_LIBRARIES}
        ${OpenCV_LIBRARIES}
     )
endif()

add_executable(featSim src/examples/feat_sim.cpp)
target_link_libraries(featSim salsa geometry ${multirotor_sim_LIBRARIES})
add_executable(gnssSim src/examples/gnss_sim.cpp)
target_link_libraries(gnssSim salsa geometry ${multirotor_sim_LIBRARIES})
add_executable(mocapSim src/examples/mocap_sim.cpp)
target_link_libraries(mocapSim salsa geometry ${multirotor_sim_LIBRARIES})
add_executable(mixedSim src/examples/mixed_sim.cpp)
target_link_libraries(mixedSim salsa geometry ${multirotor_sim_LIBRARIES})
add_executable(compareSim src/examples/compare_sim.cpp)
target_link_libraries(compareSim salsa geometry ${multirotor_sim_LIBRARIES})

