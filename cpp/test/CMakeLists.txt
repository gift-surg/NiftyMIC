#author: Michael Ebner
#date: March 2016

set( EXEC_NAME "adjointOrientedGaussianInterpolateImageFilter_UnitTests" )

#***automatically fill the PROJ_SOURCES with the sources available in the source-folder
# FILE( GLOB_RECURSE PROJ_SOURCES *.cc ) 
#***alternatively every single file:
set( PROJ_SOURCES
    "adjointOrientedGaussianInterpolateImageFilter_UnitTests.cpp"
    "../source/MyITKImageHelper.cpp"
    # ${MY_SRC_DIR}/itkAdjointOrientedGaussianInterpolateImageFilter.hxx
    )

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/bin) 


find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()

add_executable( ${EXEC_NAME} ${PROJ_SOURCES} )
target_link_libraries( ${EXEC_NAME} ${ITK_LIBRARIES} )