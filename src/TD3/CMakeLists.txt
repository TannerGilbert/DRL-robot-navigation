cmake_minimum_required(VERSION 3.0.2)
project(td3_rl)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  gazebo_ros
  gazebo_msgs
  nav_msgs
  message_generation
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs gazebo_ros gazebo_msgs nav_msgs
)

install(PROGRAMS
  scripts/train_td3.py
  scripts/test_td3.py
  scripts/gazebo_env.py
  scripts/replay_buffer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
