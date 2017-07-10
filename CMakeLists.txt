cmake_minimum_required(VERSION 2.8.3)
project(klt_tracker)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  sensor_msgs
)
find_package(OpenCV REQUIRED)


# Added "-fopenmp -O3" to make Eigen run REALLY fast
set(CMAKE_CXX_FLAGS "-std=c++0x -fopenmp -O3")

catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS roscpp rospy nav_msgs std_msgs sensor_msgs
)