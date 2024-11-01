project(texture_generator)
cmake_minimum_required(VERSION 3.0)

set( PCL_DIR ""
     CACHE FILEPATH "Folder containing \"PCLConfig.cmake\", usually \"{PCL_INSTALL_DIR}/share/pcl-x.x/\"." )

find_package(PCL REQUIRED COMPONENTS common io io_ply kdtree)
include_directories(SYSTEM ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(PNG REQUIRED)
include_directories(SYSTEM ${PNG_INCLUDE_DIR})

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Boost COMPONENTS program_options REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_STANDARD 11)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

add_executable(texturegen main.cpp generate.cpp)
target_link_libraries(texturegen
  ${PCL_LIBRARIES} ${PNG_LIBRARY} ${Boost_LIBRARIES} Eigen3::Eigen)
