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

// show 2D image
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
// show 2D image mask
void MyITKImageHelper::showImage(const MaskImageType2D::Pointer image, const std::string &filename){

  const std::string dir_output = "/tmp/";

  // Write image
  MyITKImageHelper::writeImage(image, dir_output + filename + ".nii.gz");

  // View image via itksnap
  std::string  cmd = "itksnap ";
  cmd += "-g " + dir_output + filename + ".nii.gz ";
  cmd += "& ";

  system(cmd.c_str());
}

// show 2D image + overlay
void MyITKImageHelper::showImage(const ImageType2D::Pointer image, const ImageType2D::Pointer overlay, const std::string &filename){

  const std::string dir_output = "/tmp/";

  // Write image
  MyITKImageHelper::writeImage(image, dir_output + filename + ".nii.gz");
  MyITKImageHelper::writeImage(overlay, dir_output + filename + "_overlay.nii.gz");

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

// show 2D image + mask
void MyITKImageHelper::showImage(const ImageType2D::Pointer image, const MaskImageType2D::Pointer segmentation, const std::string &filename){

  const std::string dir_output = "/tmp/";

  // Write image
  MyITKImageHelper::writeImage(image, dir_output + filename + ".nii.gz");
  MyITKImageHelper::writeImage(segmentation, dir_output + filename + "_segmentation.nii.gz");

  // View image with segmentation via itksnap
  std::string  cmd = "itksnap ";
  cmd += "-g " + dir_output + filename + ".nii.gz ";
  cmd += "-s " + dir_output + filename + "_segmentation.nii.gz ";
  cmd += "& ";

  // image with its segmentation
  // std::string  cmd = "itksnap ";
  // cmd += "-g " + dir_output + filename + ".nii.gz ";
  // cmd += "-s " + dir_output + filename + "_segmentation.nii.gz ";
  // cmd += "& ";

  system(cmd.c_str());
}

// show 3D image
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

// show 3D image mask
void MyITKImageHelper::showImage(const MaskImageType3D::Pointer image, const std::string &filename){

  const std::string dir_output = "/tmp/";

  // Write image
  MyITKImageHelper::writeImage(image, dir_output + filename + ".nii.gz");

  // View image via itksnap
  std::string  cmd = "itksnap ";
  cmd += "-g " + dir_output + filename + ".nii.gz ";
  cmd += "& ";

  system(cmd.c_str());
}

// show 3D image + overlay
void MyITKImageHelper::showImage(const ImageType3D::Pointer image, const ImageType3D::Pointer overlay, const std::string &filename){

  const std::string dir_output = "/tmp/";

  // Write image
  MyITKImageHelper::writeImage(image, dir_output + filename + ".nii.gz");
  MyITKImageHelper::writeImage(overlay, dir_output + filename + "_overlay.nii.gz");

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

// show 3D image + mask
void MyITKImageHelper::showImage(const ImageType3D::Pointer image, const MaskImageType3D::Pointer segmentation, const std::string &filename){

  const std::string dir_output = "/tmp/";

  // Write image
  MyITKImageHelper::writeImage(image, dir_output + filename + ".nii.gz");
  MyITKImageHelper::writeImage(segmentation, dir_output + filename + "_segmentation.nii.gz");

  // View image with segmentation via itksnap
  std::string  cmd = "itksnap ";
  cmd += "-g " + dir_output + filename + ".nii.gz ";
  cmd += "-s " + dir_output + filename + "_segmentation.nii.gz ";
  cmd += "& ";

  // image with its segmentation
  // std::string  cmd = "itksnap ";
  // cmd += "-g " + dir_output + filename + ".nii.gz ";
  // cmd += "-s " + dir_output + filename + "_segmentation.nii.gz ";
  // cmd += "& ";

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
void MyITKImageHelper::writeImage(const MaskImageType2D::Pointer image, const std::string &filename){

  // Create reader for nifti images
  itk::ImageFileWriter< MaskImageType2D >::Pointer writer = itk::ImageFileWriter< MaskImageType2D >::New();
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
void MyITKImageHelper::writeImage(const MaskImageType3D::Pointer image, const std::string &filename){

  // Create reader for nifti images
  itk::ImageFileWriter< MaskImageType3D >::Pointer writer = itk::ImageFileWriter< MaskImageType3D >::New();
  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
  writer->SetImageIO(imageIO);

  // Write images 
  writer->SetFileName( filename );
  writer->SetInput( image );
  writer->Update();
}