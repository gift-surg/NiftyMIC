
project(MyOrientedGaussian)

##***** Begin: Lines added by meb
# Use development mode (1) -- faster to build -- or 
# use soft launch mode (0) -- needs fully built ITK-build_dev! (which can take a while when changes are made!)
set (MY_FLAG_ITK_DEV 1)

if( ${MY_FLAG_ITK_DEV} )
    set (CMAKE_PREFIX_PATH "~/development/ITK/ITK-build/")
    set (MY_SRC_DIR "~/development/ITK/ITK/Modules/Filtering/ImageGrid/include")
    message ( "Code development mode, i.e. CMAKE_PREFIX_PATH = " ${CMAKE_PREFIX_PATH} )
else( ${MY_FLAG_ITK_DEV} )
    set (CMAKE_PREFIX_PATH "~/development/ITK/ITK-build_dev/")
    message ( "Soft launch mode, i.e. CMAKE_PREFIX_PATH = " ${CMAKE_PREFIX_PATH} )
endif( ${MY_FLAG_ITK_DEV} )

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 
set (CMAKE_BUILD_TYPE "Debug")


##***** End: Lines added by meb

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()
 
# add_executable(adjointOrientedGaussianInterpolateImageFilter adjointOrientedGaussianInterpolateImageFilter.cpp MyITKImageHelper.cpp ${MY_SRC_DIR}/itkAdjointOrientedGaussianInterpolateImageFilter.hxx)
# add_executable(adjointOrientedGaussianInterpolateImageFilter adjointOrientedGaussianInterpolateImageFilter.cpp MyITKImageHelper.cpp)
# target_link_libraries(adjointOrientedGaussianInterpolateImageFilter ITKReview ${ITK_LIBRARIES})


add_executable(resampleImageFilter resampleImageFilter.cpp)

# if( ${MY_FLAG_ITK_DEV} )
#     add_executable(adjointOrientedGaussianInterpolateImageFilter adjointOrientedGaussianInterpolateImageFilter.cpp MyITKImageHelper.cpp ${MY_SRC_DIR}/itkAdjointOrientedGaussianInterpolateImageFilter.hxx)
# else( ${MY_FLAG_ITK_DEV} )
#     add_executable(adjointOrientedGaussianInterpolateImageFilter adjointOrientedGaussianInterpolateImageFilter.cpp MyITKImageHelper.cpp)
# endif( ${MY_FLAG_ITK_DEV} )

 
if( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(resampleImageFilter ITKReview ${ITK_LIBRARIES})
  # target_link_libraries(adjointOrientedGaussianInterpolateImageFilter ITKReview ${ITK_LIBRARIES})
  # target_link_libraries(itkOrientedGaussianInterpolateImageFunction ITKReview ${ITK_LIBRARIES})
else( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(resampleImageFilter ${ITK_LIBRARIES})
  # target_link_libraries(adjointOrientedGaussianInterpolateImageFilter ${ITK_LIBRARIES})
  # target_link_libraries(itkOrientedGaussianInterpolateImageFunction ${ITK_LIBRARIES})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )