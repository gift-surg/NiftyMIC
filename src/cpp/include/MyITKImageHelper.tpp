/*! \brief Code to verify the implementation of itkAdjointOrientedGaussianInterpolateImageFilter.
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date February 2016
 */

#include "MyITKImageHelper.h"

/** Read image */
template <typename ImageType>
const typename ImageType::Pointer 
MyITKImageHelper::readImage(const std::string &filename){

  // Create reader for nifti images
  typename itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New();
  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  

  // Parametrize reader
  reader->SetImageIO(imageIO);
  reader->SetFileName(filename);
  reader->Update();

  typename ImageType::Pointer image = reader->GetOutput();
  image->DisconnectPipeline();
  return image;
}
