cmake_minimum_required(VERSION 3.0.2)
project(store_monitoring_messages)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(MSG_DEPS
  std_msgs
  geometry_msgs
)

# set(MSG_FILES
#   ClusterSeparators.srv
#  )

set(SRV_FILES
  ClusterSeparators.srv
)

# set(ACT_FILES
#   Move.action
# )

find_package(catkin REQUIRED COMPONENTS
  ${MSG_DEPS}
  message_generation
)

# add_message_files(DIRECTORY msg FILES ${MSG_FILES})
add_service_files(DIRECTORY srv FILES ${SRV_FILES})
# add_action_files(DIRECTORY action FILES ${ACT_FILES})

generate_messages(DEPENDENCIES ${MSG_DEPS})

catkin_package(DEPENDS ${MSG_DEPS} message_runtime)
