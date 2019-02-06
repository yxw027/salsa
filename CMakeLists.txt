cmake_minimum_required(VERSION 2.8.3)
project(salsa)

set(CMAKE_CXX_STANDARD 14)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest)
find_package(yaml-cpp REQUIRED)
find_package(catkin COMPONENTS
	rosbag
	roscpp
	sensor_msgs
	nav_msgs)

add_subdirectory(lib/geometry)
add_subdirectory(lib/multirotor_sim)

include_directories(include
    lib
    lib/multirotor_sim/include
    lib/geometry/include
    lib/utils/include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR})

add_library(salsa
    src/salsa.cpp
    src/num_diff.cpp)
target_link_libraries(salsa
    geometry
    pthread
    multirotor_sim
    ${YAML_CPP_LIBRARIES}
    ${CERES_LIBRARIES})


if (${catkin_FOUND})
    message(STATUS "catkin_FOUND - Building salsa ROS library")
	include_directories(${catkin_INCLUDE_DIRS})
	add_library(salsa_ros src/salsa_ros.cpp)
    target_link_libraries(salsa_ros salsa ${catkin_LIBRARIES})
	add_executable(salsa_rosbag src/salsa_rosbag.cpp)
    target_link_libraries(salsa_rosbag salsa_ros ${catkin_LIBRARIES})
	add_executable(salsa_node src/salsa_node.cpp)
	target_link_libraries(salsa_node salsa_ros ${catkin_LIBRARIES})
endif()


if (${GTEST_FOUND})
    message(STATUS "GTEST_FOUND - Building salsa tests")
	add_executable(test_salsa
		src/test/test_salsa.cpp
        src/test/test_salsa_mocap_indexing.cpp
        src/test/test_salsa_gps_indexing.cpp
        src/test/test_mocap_factor.cpp
        src/test/test_pseudorange_factor.cpp
        src/test/test_imu_factor.cpp
        src/test/test_logger.cpp
        src/test/test_point_positioning.cpp
		src/test/test_misc.cpp)
	target_link_libraries(test_salsa
        salsa
		geometry
		multirotor_sim
		gtest
		gtest_main
        pthread
		${GTEST_LIBRARIES}
		${CERES_LIBRARIES})
endif()

