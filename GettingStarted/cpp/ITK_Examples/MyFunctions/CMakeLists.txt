cmake_minimum_required(VERSION 2.8)
 
project(ResampleImageFilter)

## Begin: Lines added by meb
set (CMAKE_PREFIX_PATH "~/development/ITK/ITK-build_dev/")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 
set (CMAKE_BUILD_TYPE "Debug")
## End: Lines added by meb

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()
 
add_executable(ResampleImageFilter ResampleImageFilter.cxx)
 
if( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(ResampleImageFilter ITKReview ${ITK_LIBRARIES})
else( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(ResampleImageFilter ${ITK_LIBRARIES})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )