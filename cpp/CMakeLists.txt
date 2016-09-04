#author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#date:   March 2016

cmake_minimum_required(VERSION 2.8)
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


set (MY_BUILD_TYPE "Release") #Release, Debug, RelWithDebInfo, MinSizeRel


##############################################################################
# CMAKE_BUILD_TYPE
##############################################################################
# \note: A bit bulky but simple set (CMAKE_BUILD_TYPE "Release") would not work!

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${MY_BUILD_TYPE}' as none was specified.")
  set (CMAKE_BUILD_TYPE ${MY_BUILD_TYPE} CACHE STRING "Choose the type of build." FORCE)

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()


##############################################################################
# Find Boost.
##############################################################################
set(Boost_NO_SYSTEM_PATHS ON)
if (Boost_NO_SYSTEM_PATHS)
  if(APPLE)
    set(BOOST_ROOT "/opt/local/include/")   #Mac
    message("Mac OS X: Set BOOS_ROOT=${BOOST_ROOT}")
  elseif(UNIX)
    set(BOOST_ROOT "/usr/include/")       #Ubuntu
    message("Linux: Set BOOS_ROOT=${BOOST_ROOT}")
  endif()
endif (Boost_NO_SYSTEM_PATHS)
if(BUILD_SHARED_LIBS)
  set(Boost_USE_STATIC_LIBS OFF)
else()
  set(Boost_USE_STATIC_LIBS ON)
endif()
find_package( Boost COMPONENTS filesystem system date_time regex thread 
  iostreams program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

##############################################################################
# Actual project.
##############################################################################
# Some important CMAKE variables:
# () CMAKE_SOURCE_DIR refers to the root source directory containing (this) 
#    CMakeLists.txt file
# () CMAKE_BINARY_DIR points to the current build directory

include_directories(include)
include_directories(test/catch)

#***Specify location for all generated executables:
# set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 

#***Directories with additional CMakeLists.txt:
add_subdirectory(source)
add_subdirectory(test)


# IF(CMAKE_BUILD_TYPE MATCHES Debug)
#   message("Debug build.")
# ELSEIF(CMAKE_BUILD_TYPE MATCHES Release)
#   message("Release build.")
# ELSE()
#   message("Some other build type.")
# ENDIF()

##############################################################################
# Copy test data into build folder.
##############################################################################

file(GLOB MY_COPY_FILES
  "test/data/*"
)
file(COPY ${MY_COPY_FILES} DESTINATION ${CMAKE_BINARY_DIR}/test/exampleData)