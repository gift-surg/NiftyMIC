/*! \brief Code to verify the implementation of itkAdjointOrientedGaussianInterpolateImageFilter.
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date February 2016
 */

#include "MyITKImageHelper.h"

/** Constructor */
// template <typename ImageType>
// MyITKImageHelper::MyITKImageHelper(){}

/** Destructor */
// template <typename ImageType>
// MyITKImageHelper::~MyITKImageHelper(){}

/** Show image */
void MyITKImageHelper::showImage(const ImageType2D::Pointer image, const std::string &filename){

  const std::string dir_output = "/tmp/";

  // Write image
  MyITKImageHelper::writeImage(image, dir_output + filename + ".nii.gz");

  // View image via itksnap
  std::string  cmd = "itksnap ";
  cmd += "-g " + dir_output + filename + ".nii.gz ";
  cmd += "& ";

  system(cmd.c_str());
}

void MyITKImageHelper::showImage(const ImageType2D::Pointer image, const ImageType2D::Pointer overlay, const std::string &filename){

  const std::string dir_output = "/tmp/";

  // Write image
  MyITKImageHelper::writeImage(image, dir_output + filename + ".nii.gz");
  MyITKImageHelper::writeImage(image, dir_output + filename + "_overlay.nii.gz");

  // View image with overlay via itksnap
  std::string  cmd = "itksnap ";
  cmd += "-g " + dir_output + filename + ".nii.gz ";
  cmd += "-o " + dir_output + filename + "_overlay.nii.gz ";
  cmd += "& ";

  // image with its segmentation
  // std::string  cmd = "itksnap ";
  // cmd += "-g " + dir_output + filename + ".nii.gz ";
  // cmd += "-s " + dir_output + filename + "_segmentation.nii.gz ";
  // cmd += "& ";

  system(cmd.c_str());
}

void MyITKImageHelper::showImage(const ImageType3D::Pointer image, const std::string &filename){

  const std::string dir_output = "/tmp/";

  // Write image
  MyITKImageHelper::writeImage(image, dir_output + filename + ".nii.gz");

  // View image via itksnap
  std::string  cmd = "itksnap ";
  cmd += "-g " + dir_output + filename + ".nii.gz ";
  cmd += "& ";

  system(cmd.c_str());
}

/** Write image */
void MyITKImageHelper::writeImage(const ImageType2D::Pointer image, const std::string &filename){

  // Create reader for nifti images
  itk::ImageFileWriter< ImageType2D >::Pointer writer = itk::ImageFileWriter< ImageType2D >::New();
  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
  writer->SetImageIO(imageIO);

  // Write images
  writer->SetFileName( filename );
  writer->SetInput( image );
  writer->Update();
}
void MyITKImageHelper::writeImage(const ImageType3D::Pointer image, const std::string &filename){

  // Create reader for nifti images
  itk::ImageFileWriter< ImageType3D >::Pointer writer = itk::ImageFileWriter< ImageType3D >::New();
  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
  writer->SetImageIO(imageIO);

  // Write images 
  writer->SetFileName( filename );
  writer->SetInput( image );
  writer->Update();
}