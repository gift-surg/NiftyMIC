 
 
## Begin: Lines added by meb
set (MY_PROJ_NAME "DoDataParallelThreading")
set (MY_MAIN_FILE "main.cpp")
set (MY_EXEC_FILE "doDataParallelThreading")
set (CMAKE_PREFIX_PATH "~/development/ITK/ITK-build/")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 
## End: Lines added by meb

cmake_minimum_required(VERSION 2.8)
project(${MY_PROJ_NAME})

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()
 
add_executable( ${MY_EXEC_FILE} ${MY_MAIN_FILE} )
 
if( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(${MY_EXEC_FILE} ITKReview ${ITK_LIBRARIES})
else( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(${MY_EXEC_FILE} ${ITK_LIBRARIES})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )