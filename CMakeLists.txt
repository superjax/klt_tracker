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

catkin_python_setup()

# catkin_install_python(PROGRAMS src/klt_tracker.py
#                       DESTINATION ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})
#                       # Put an __init__.py file in the package location so it can be imported
# file(WRITE ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/__init__.py "")
