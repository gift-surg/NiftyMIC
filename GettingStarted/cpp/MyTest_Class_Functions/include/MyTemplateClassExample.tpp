/*! \brief Code to verify the implementation of itkAdjointOrientedGaussianInterpolateImageFilter.
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date February 2016
 */

#include "MyITKImageHelper.h"


/** Constructor */
template <typename ImageType>
MyITKImageHelper<ImageType>::MyITKImageHelper(){}

/** Destructor */
template <typename ImageType>
MyITKImageHelper<ImageType>::~MyITKImageHelper(){}


/** Show image */
template <typename ImageType>
void MyITKImageHelper<ImageType>::showImage(const typename ImageType::Pointer image){

  const std::string dir_output = "/tmp/";
  const std::string filename_output = "test.nii.gz";

  typename itk::ImageFileWriter< ImageType >::Pointer writer = itk::ImageFileWriter< ImageType >::New();

  /* Write images */
  writer->SetFileName( dir_output + filename_output );
  writer->SetInput( image );
  writer->Update();

  /* View image via itksnap */
  std::string  cmd = "itksnap ";
  cmd += "-g " + dir_output + filename_output;
  cmd += "& ";

  system(cmd.c_str());
}


/** Read image */
template <typename ImageType>
const typename ImageType::Pointer 
MyITKImageHelper<ImageType>::readImage(const std::string &filename){
  // Create reader
  typename itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New();

  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
  reader->SetImageIO(imageIO);
  reader->SetFileName(filename);
  reader->Update();

  // const typename ImageType::Pointer im = ;
  // std::cout << im->GetSpacing() << std::endl;

  return reader->GetOutput();

}
