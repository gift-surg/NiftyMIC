#author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#data:   February 2016

cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

##############################################################################
# Actual project.
##############################################################################
# Some important CMAKE variables:
# () CMAKE_SOURCE_DIR refers to the root source directory containing (this) 
#    CMakeLists.txt file
# () CMAKE_BINARY_DIR points to the current build directory

# include_directories(include)
# include_directories(test/catch)

#***Specify location for all generated executables:
# set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 

#***Directories with additional CMakeLists.txt:
add_subdirectory(source)
# add_subdirectory(test)

##############################################################################
# Copy test data into build folder.
##############################################################################

# file(GLOB MY_COPY_FILES
#   "test/data/*"
# )
# file(COPY ${MY_COPY_FILES} DESTINATION ${CMAKE_BINARY_DIR}/test/exampleData)