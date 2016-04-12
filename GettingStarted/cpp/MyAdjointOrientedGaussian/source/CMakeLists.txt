#author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#date:   February 2016

#---- Define variables ------------------------------
## Location of ITK-build folder
# set (CMAKE_PREFIX_PATH "~/development/ITK/ITK-build_dev/")
set (CMAKE_PREFIX_PATH "~/development/ITK/ITK-build/")

## Location of itkAdjointOrientedGaussianInterpolateImageFilter.h and hxx
set (MY_SRC_DIR "~/development/ITK/ITK/Modules/Filtering/ImageGrid/include")

## Project specific settings
set ( MY_PROJ_NAME "AdjointOrientedGaussian" )

set ( MY_MAIN_FILE_1 "adjointOrientedGaussianInterpolateImageFilter.cpp" )
set ( MY_EXEC_FILE_1 "adjointOrientedGaussianInterpolateImageFilter" )
set ( MY_MAIN_FILE_1_SOURCES 
    "MyITKImageHelper.cpp" 
    # "${MY_SRC_DIR}/itkAdjointOrientedGaussianInterpolateImageFilter.hxx"
    )

set ( MY_MAIN_FILE_2 "doDataParallelThreading.cpp" )
set ( MY_EXEC_FILE_2 "doDataParallelThreading" )
set ( MY_MAIN_FILE_2_SOURCES 
    # "MyITKImageHelper.cpp" 
    # "${MY_SRC_DIR}/itkAdjointOrientedGaussianInterpolateImageFilter.hxx"
    )

message (STATUS "Use ITK-directory " ${CMAKE_PREFIX_PATH})


project( ${MY_PROJ_NAME} )

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 


find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()
 


add_executable( ${MY_EXEC_FILE_1} ${MY_MAIN_FILE_1} ${MY_MAIN_FILE_1_SOURCES} )  
target_link_libraries( ${MY_EXEC_FILE_1} ${ITK_LIBRARIES} )

add_executable( ${MY_EXEC_FILE_2} ${MY_MAIN_FILE_2} ${MY_MAIN_FILE_2_SOURCES} )  
target_link_libraries( ${MY_EXEC_FILE_2} ${ITK_LIBRARIES} )


