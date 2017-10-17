#author: Michael Ebner
#date: March 2016


#***automatically fill the PROJ_SOURCES with the sources available in the source-folder
# FILE( GLOB_RECURSE PROJ_SOURCES *.cc ) 
#***alternatively every single file:

set( EXEC_NAME "itkOrientedGaussianInterpolateImageFilter_UnitTests" )
set( PROJ_SOURCES
    "itkOrientedGaussianInterpolateImageFilter_UnitTests.cpp"
    "../source/MyITKImageHelper.cpp"
    "../source/MyException.cpp"
    )

set( EXEC_NAME_2 "itkAdjointOrientedGaussianInterpolateImageFilter_UnitTests" )
set( PROJ_SOURCES_2
    "itkAdjointOrientedGaussianInterpolateImageFilter_UnitTests.cpp"
    "../source/MyITKImageHelper.cpp"
    "../source/MyException.cpp"
    )

set( EXEC_NAME_3 "itkInplaneSimilarity3DTransform_UnitTests" )
set( PROJ_SOURCES_3
    "itkInplaneSimilarity3DTransform_UnitTests.cpp"
    "../source/MyITKImageHelper.cpp"
    "../source/MyException.cpp"
    )

set( EXEC_NAME_4 "itkGradientEuler3DTransformImageFilter_UnitTests" )
set( PROJ_SOURCES_4
    "itkGradientEuler3DTransformImageFilter_UnitTests.cpp"
    "../source/MyITKImageHelper.cpp"
    "../source/MyException.cpp"
    )

set( EXEC_NAME_5 "itkGradientAffine3DTransformImageFilter_UnitTests" )
set( PROJ_SOURCES_5
    "itkGradientAffine3DTransformImageFilter_UnitTests.cpp"
    "../source/MyITKImageHelper.cpp"
    "../source/MyException.cpp"
    )

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/bin) 

add_executable( ${EXEC_NAME} ${PROJ_SOURCES} )
target_link_libraries( ${EXEC_NAME} ${ITK_LIBRARIES} )

add_executable( ${EXEC_NAME_2} ${PROJ_SOURCES_2} )
target_link_libraries( ${EXEC_NAME_2} ${ITK_LIBRARIES} )

add_executable( ${EXEC_NAME_3} ${PROJ_SOURCES_3} )
target_link_libraries( ${EXEC_NAME_3} ${ITK_LIBRARIES} )

add_executable( ${EXEC_NAME_4} ${PROJ_SOURCES_4} )
target_link_libraries( ${EXEC_NAME_4} ${ITK_LIBRARIES} )

add_executable( ${EXEC_NAME_5} ${PROJ_SOURCES_5} )
target_link_libraries( ${EXEC_NAME_5} ${ITK_LIBRARIES} )