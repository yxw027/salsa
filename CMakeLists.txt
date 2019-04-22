cmake_minimum_required(VERSION 2.8.3)
project(salsa)

set(CMAKE_CXX_STANDARD 17)

if(${SALSA_WINDOW_SIZE})
    add_definitions(-DSALSA_WINDOW_SIZE=${SALSA_WINDOW_SIZE})
    message("Defining Custom Window Size: ${SALSA_WINDOW_SIZE}")
endif()

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest)
find_package(yaml-cpp REQUIRED)
find_package(catkin REQUIRED COMPONENTS
	rosbag
	roscpp
	sensor_msgs
        inertial_sense
	nav_msgs)

add_subdirectory(lib/geometry)
add_subdirectory(lib/gnss_utils)
add_subdirectory(lib/multirotor_sim)

add_definitions(-DMULTIROTOR_SIM_DIR="${CMAKE_CURRENT_LIST_DIR}/lib/multirotor_sim")
add_definitions(-DSALSA_DIR="${CMAKE_CURRENT_LIST_DIR}")

add_library(factors
    src/factors/xform.cpp
    src/factors/imu.cpp
    src/factors/mocap.cpp
    src/factors/feat.cpp
    src/factors/anchor.cpp
    src/factors/clock_dynamics.cpp
    src/factors/pseudorange.cpp)
target_link_libraries(factors
    geometry
    gnss_utils
    ${CERES_LIBRARIES})
target_include_directories(factors PUBLIC
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS})

add_library(salsa
    src/state.cpp
    src/salsa.cpp
    src/salsa_mocap.cpp
    src/salsa_gnss.cpp
    src/salsa_vision.cpp
    src/salsa_log.cpp
    src/klt.cpp
    src/num_diff.cpp)
target_link_libraries(salsa
    geometry
    pthread
    multirotor_sim
    factors
    stdc++fs
    ${YAML_CPP_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES})


if (${catkin_FOUND})
    message(STATUS "catkin_FOUND - Building salsa ROS library")
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


if (${GTEST_FOUND})
    message(STATUS "GTEST_FOUND - Building salsa tests")
    add_executable(test_salsa
#        src/test/test_salsa.cpp
        src/test/test_mocap_factor.cpp
        src/test/test_imu_factor.cpp
        src/test/test_logger.cpp
        src/test/test_point_positioning.cpp
        src/test/test_feat_factor.cpp
        src/test/test_misc.cpp
        src/test/test_vision.cpp
        src/test/test_klt.cpp
#        src/test/test_salsa_mocap_indexing.cpp
#        src/test/test_pseudorange_factor.cpp
#        src/test/test_salsa_gps_indexing.cpp
     )
     target_link_libraries(test_salsa
         salsa
         gtest
         gtest_main
         multirotor_sim
         ${GTEST_LIBRARIES}
     )
endif()

add_executable(featSim src/examples/feat_sim.cpp)
target_link_libraries(featSim salsa geometry multirotor_sim)
add_executable(gnssSim src/examples/gnss_sim.cpp)
target_link_libraries(gnssSim salsa geometry multirotor_sim)
add_executable(mocapSim src/examples/mocap_sim.cpp)
target_link_libraries(mocapSim salsa geometry multirotor_sim)

