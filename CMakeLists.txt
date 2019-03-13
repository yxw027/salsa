cmake_minimum_required(VERSION 2.8.3)
project(salsa)

set(CMAKE_CXX_STANDARD 14)
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if(${SALSA_WINDOW_SIZE})
    add_definitions(-DSALSA_WINDOW_SIZE=${SALSA_WINDOW_SIZE})
    message("Defining Custom Window Size: ${SALSA_WINDOW_SIZE}")
endif()

include(cotire)
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
    lib/utils/include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR})

add_library(factors
    src/factors/xform.cpp
    src/factors/imu.cpp
    src/factors/mocap.cpp
    src/factors/feat.cpp
    src/factors/anchor.cpp
    src/factors/pseudorange.cpp)
target_link_libraries(factors
    geometry
    multirotor_sim
    ${CERES_LIBRARIES})

add_library(salsa
    src/salsa.cpp
    src/num_diff.cpp)
target_link_libraries(salsa
    geometry
    pthread
    multirotor_sim
    factors
    stdc++fs
    ${YAML_CPP_LIBRARIES}
    ${CERES_LIBRARIES})
cotire(salsa)


if (${catkin_FOUND})
    message(STATUS "catkin_FOUND - Building salsa ROS library")
	add_library(salsa_ros src/salsa_ros.cpp)
    target_include_directories(salsa_ros PUBLIC ${catkin_INCLUDE_DIRS})
    target_link_libraries(salsa_ros salsa ${catkin_LIBRARIES})
    cotire(salsa_ros)

	add_executable(salsa_rosbag src/salsa_rosbag.cpp)
    target_include_directories(salsa_rosbag PUBLIC ${catkin_INCLUDE_DIRS})
    target_link_libraries(salsa_rosbag salsa_ros ${catkin_LIBRARIES})
    cotire(salsa_rosbag)

	add_executable(salsa_node src/salsa_node.cpp)
	target_link_libraries(salsa_node salsa_ros ${catkin_LIBRARIES})
    target_include_directories(salsa_node PUBLIC ${catkin_INCLUDE_DIRS})
    cotire(salsa_node)
endif()


if (${GTEST_FOUND})
    message(STATUS "GTEST_FOUND - Building salsa tests")
    function(make_gtest test_name source_file)
        add_executable(${test_name} ${source_file})
        target_link_libraries(${test_name}
            salsa
            geometry
            gtest
            gtest_main
            multirotor_sim
            ${GTEST_LIBRARIES}
            ${CERES_LIBRARIES}
            )
        cotire(${test_name})
    endfunction()

	make_gtest(test_salsa src/test/test_salsa.cpp)
#	make_gtest(test_salsa_mocap_indexing src/test/test_salsa_mocap_indexing.cpp)
#	make_gtest(test_salsa_gps_indexing src/test/test_salsa_gps_indexing.cpp)
#	make_gtest(test_mocap_factor src/test/test_mocap_factor.cpp)
#	make_gtest(test_pseudorange_factor src/test/test_pseudorange_factor.cpp)
#	make_gtest(test_imu_factor src/test/test_imu_factor.cpp)
#	make_gtest(test_logger src/test/test_logger.cpp)
#	make_gtest(test_point_positioning src/test/test_point_positioning.cpp)
#    make_gtest(test_misc src/test/test_misc.cpp)
endif()

