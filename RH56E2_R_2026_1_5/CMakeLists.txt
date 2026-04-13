cmake_minimum_required(VERSION 3.5)

project(RH56E2_R_2026_1_5)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)
  
install(DIRECTORY meshes/
  DESTINATION share/${PROJECT_NAME}/meshes)
  
install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz)

  
ament_package()



