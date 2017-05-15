#author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#date:   May 2016

#---- Define variables ------------------------------
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

set ( MY_MAIN_FILE_2 "runN4BiasFieldCorrectionImageFilter.cpp" )
set ( MY_EXEC_FILE_2 "runN4BiasFieldCorrectionImageFilter" )
set ( MY_MAIN_FILE_SOURCES_2 
    "MyITKImageHelper.cpp" 
    "readCommandLine.cpp"
    "MyException.cpp"
    "ExceptionCommandLine.cpp"
    )

set ( MY_MAIN_FILE_3 "itkInplaneSimilarity3DReg.cpp" )
set ( MY_EXEC_FILE_3 "itkInplaneSimilarity3DReg" )
set ( MY_MAIN_FILE_SOURCES_3
    "MyITKImageHelper.cpp" 
    "readCommandLine.cpp"
    "MyException.cpp"
    "ExceptionCommandLine.cpp"
    )

project( ${MY_PROJ_NAME} )

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 

add_executable( ${MY_EXEC_FILE} ${MY_MAIN_FILE} ${MY_MAIN_FILE_SOURCES} )  
target_link_libraries( ${MY_EXEC_FILE} ${ITK_LIBRARIES} )
target_link_libraries( ${MY_EXEC_FILE} ${Boost_LIBRARIES} )

add_executable( ${MY_EXEC_FILE_2} ${MY_MAIN_FILE_2} ${MY_MAIN_FILE_SOURCES_2} )  
target_link_libraries( ${MY_EXEC_FILE_2} ${ITK_LIBRARIES} )
target_link_libraries( ${MY_EXEC_FILE_2} ${Boost_LIBRARIES} )

add_executable( ${MY_EXEC_FILE_3} ${MY_MAIN_FILE_3} ${MY_MAIN_FILE_SOURCES_3} )  
target_link_libraries( ${MY_EXEC_FILE_3} ${ITK_LIBRARIES} )
target_link_libraries( ${MY_EXEC_FILE_3} ${Boost_LIBRARIES} )