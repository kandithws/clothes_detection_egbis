cmake_minimum_required(VERSION 2.8.3)
project(egbis)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        cmake_modules
        )

# find_package(cmake_modules REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

# include_directories(include/${PROJECT_NAME})
# include_directories(include/${PROJECT_NAME})

set(EGBIS_LIB_SRC lib/segment-image.cpp lib/segment-graph.cpp lib/disjoint-set.cpp lib/filter.cpp lib/misc.cpp)
# set(EGBIS_H egbis/segment-image.h egbis/segment-graph.h egbis/disjoint-set.h egbis/image.h egbis/filter.h egbis/misc.h)
catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
)

add_library(${PROJECT_NAME} ${EGBIS_LIB_SRC})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

install(TARGETS egbis ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h"
        PATTERN ".svn" EXCLUDE)

