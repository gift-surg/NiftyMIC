#author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#date:   May 2016

#---- Define variables ------------------------------
## Location of ITK-build folder
set (CMAKE_PREFIX_PATH "~/development/ITK/ITK-build_dev/")
# set (ITK_DIR "~/development/ITK/ITK-build_dev/")
# set (ITK_DIR "~/development/ITK/ITK-build_dev/" CACHE STRING "Choose the type of build." FORCE)

## Location of itkAdjointOrientedGaussianInterpolateImageFilter.h and hxx
# set (MY_SRC_DIR "~/development/ITK/ITK/Modules/Filtering/ImageGrid/include")

## Project specific settings
set ( MY_PROJ_NAME "itkRegistrationMethods" )

set ( MY_MAIN_FILE "itkReg.cpp" )
set ( MY_EXEC_FILE "itkReg" )
set ( MY_MAIN_FILE_SOURCES 
    "MyITKImageHelper.cpp" 
    "readCommandLine.cpp"
    "MyException.cpp"
    "ExceptionCommandLine.cpp"
    )

set ( MY_MAIN_FILE_2 "itkRigidReg.cpp" )
set ( MY_EXEC_FILE_2 "itkRigidReg" )
set ( MY_MAIN_FILE_SOURCES_2 
    "MyITKImageHelper.cpp" 
    )

# message (STATUS "Use ITK-directory " ${ITK_DIR})

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()
# message (STATUS "Use ITK-directory " ${ITK_DIR})

project( ${MY_PROJ_NAME} )

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 

add_executable( ${MY_EXEC_FILE} ${MY_MAIN_FILE} ${MY_MAIN_FILE_SOURCES} )  
add_executable( ${MY_EXEC_FILE_2} ${MY_MAIN_FILE_2} ${MY_MAIN_FILE_SOURCES_2} )  

target_link_libraries( ${MY_EXEC_FILE} ${ITK_LIBRARIES} )
target_link_libraries( ${MY_EXEC_FILE} ${Boost_LIBRARIES} )
target_link_libraries( ${MY_EXEC_FILE_2} ${ITK_LIBRARIES} )


