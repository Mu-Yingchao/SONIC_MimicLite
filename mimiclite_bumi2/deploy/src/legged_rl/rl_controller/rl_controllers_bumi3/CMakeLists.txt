cmake_minimum_required(VERSION 3.10)
project(rl_controllers_bumi3)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  controller_interface
  controller_manager_msgs
  dynamic_reconfigure
  gazebo_msgs
  kdl_parser
  legged_common
  message_generation
  robot_state_publisher
  roscpp
  sensor_msgs
  std_msgs
  tf
)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    controller_interface
    controller_manager_msgs
    dynamic_reconfigure
    gazebo_msgs
    kdl_parser
    legged_common
    robot_state_publisher
    roscpp
    sensor_msgs
    std_msgs
    tf
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/RLControllerBase.cpp
  src/AcController.cpp
  src/utilities.cpp
  src/log_recorder.cpp
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME QUIET libonnxruntime)
if(NOT ONNXRUNTIME_FOUND)
  find_package(onnxruntime REQUIRED)
endif()

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  onnxruntime
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/rl_controllers_bumi3/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
install(FILES rl_controllers_bumi3_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY config launch motion_files policy rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
