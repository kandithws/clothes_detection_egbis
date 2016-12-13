cmake_minimum_required(VERSION 2.8.3)
project(clothes_detection)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        actionlib_msgs
        cv_bridge
        geometry_msgs
        image_transport
        pcl_conversions
        pcl_ros
        roscpp
        sensor_msgs
        std_msgs
        message_generation
        egbis
        )

find_package(OpenCV 2 REQUIRED)
find_package(egbis REQUIRED)

add_message_files(
        FILES
        Clothes.msg
        ClothesArray.msg
)

add_action_files(
        DIRECTORY action
        FILES FindClothes.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
        DEPENDENCIES
        geometry_msgs
        std_msgs
        sensor_msgs
        actionlib_msgs
)

include_directories(
        ${catkin_INCLUDE_DIRS}
        include/${PROJECT_NAME}
)

include_directories(${egbis_INCLUDE_DIRS})

catkin_package(
        CATKIN_DEPENDS egbis roscpp std_msgs sensor_msgs pcl_ros pcl_conversions
        geometry_msgs actionlib_msgs message_runtime
        #  DEPENDS system_lib
        # LIBRARIES egbis
)



add_executable(clothes_detection_node src/clothes_detection_node.cpp src/egbis.cpp src/ClothesDetector.cpp)
add_dependencies(clothes_detection_node ${catkin_EXPORTED_TARGETS} egbis)
target_link_libraries(clothes_detection_node egbis ${catkin_LIBRARIES} ${pcl_ros_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})


