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

void MyITKImageHelper::printTransform(itk::AffineTransform< PixelType, 3 >::ConstPointer transform){
  
  const unsigned int dim = 3;

  itk::Matrix< PixelType, dim, dim >  matrix = transform->GetMatrix();

  itk::AffineTransform< PixelType, dim >::ParametersType parameters = transform->GetParameters();
  itk::AffineTransform< PixelType, dim >::ParametersType center = transform->GetFixedParameters();

  std::cout << "Transform:" << std::endl;

  std::cout << "\t center = " << std::endl;
  printf("\t\t%.4f\t%.4f\t%.4f\n", center[0], center[1], center[2]);
  
  std::cout << "\t translation = " << std::endl;
  printf("\t\t%.4f\t%.4f\t%.4f\n", parameters[3], parameters[4], parameters[5]);
  
  std::cout << "\t matrix = " << std::endl;
  for (int i = 0; i < dim; ++i) {
    printf("\t\t%.4f\t%.4f\t%.4f\n", matrix[i][0], matrix[i][1], matrix[i][2]);
  }
}

void MyITKImageHelper::printTransform(itk::Euler3DTransform< PixelType >::ConstPointer transform){
  
  const unsigned int dim = 3;

  itk::Matrix< PixelType, dim, dim >  matrix = transform->GetMatrix();

  itk::Euler3DTransform< PixelType >::ParametersType parameters = transform->GetParameters();
  itk::Euler3DTransform< PixelType >::ParametersType center = transform->GetFixedParameters();

  std::cout << "Transform:" << std::endl;

  std::cout << "\t center = " << std::endl;
  printf("\t\t%.4f\t%.4f\t%.4f\n", center[0], center[1], center[2]);

  std::cout << "\t angle_x_deg, angle_y_deg, angle_z_deg = " << std::endl;
  printf("\t\t%.4f, %.4f, %.4f\n", parameters[0]*180/vnl_math::pi, parameters[1]*180/vnl_math::pi, parameters[2]*180/vnl_math::pi);
  
  std::cout << "\t translation = " << std::endl;
  printf("\t\t%.4f\t%.4f\t%.4f\n", parameters[3], parameters[4], parameters[5]);
  
  std::cout << "\t matrix = " << std::endl;
  for (int i = 0; i < dim; ++i) {
    printf("\t\t%.4f\t%.4f\t%.4f\n", matrix[i][0], matrix[i][1], matrix[i][2]);
  }
}

void MyITKImageHelper::printTransform(itk::ScaledTranslationEuler3DTransform< PixelType >::ConstPointer transform){
  
  const unsigned int dim = 3;

  itk::Matrix< PixelType, dim, dim >  matrix = transform->GetMatrix();

  itk::ScaledTranslationEuler3DTransform< PixelType >::ParametersType parameters = transform->GetParameters();
  itk::ScaledTranslationEuler3DTransform< PixelType >::ParametersType center = transform->GetFixedParameters();

  std::cout << "Transform:" << std::endl;

  std::cout << "\t center = " << std::endl;
  printf("\t\t%.4f\t%.4f\t%.4f\n", center[0], center[1], center[2]);

  std::cout << "\t angle_x_deg, angle_y_deg, angle_z_deg = " << std::endl;
  printf("\t\t%.4f, %.4f, %.4f\n", parameters[0]*180/vnl_math::pi, parameters[1]*180/vnl_math::pi, parameters[2]*180/vnl_math::pi);
  
  std::cout << "\t translation = " << std::endl;
  printf("\t\t%.4f\t%.4f\t%.4f\n", parameters[3], parameters[4], parameters[5]);
  
  std::cout << "\t matrix = " << std::endl;
  for (int i = 0; i < dim; ++i) {
    printf("\t\t%.4f\t%.4f\t%.4f\n", matrix[i][0], matrix[i][1], matrix[i][2]);
  }
}

void MyITKImageHelper::writeTransform(itk::AffineTransform< PixelType, 3 >::ConstPointer transform,
    std::string outfile){

    const unsigned int dim = 3;

    itk::AffineTransform< PixelType, dim >::ParametersType parameters = transform->GetParameters();
    itk::AffineTransform< PixelType, dim >::ParametersType fixedParameters = transform->GetFixedParameters();

    const unsigned int iNumberOfParameters = parameters.size();
    const unsigned int iNumberOfFixedParameters = fixedParameters.size();

    std::ofstream output(outfile);

    if(!output.is_open()){
        throw MyException("Cannot open the file to write");
    }

    // Define output precision
    output.precision(10); 

    // output << "iteration \t" << "elapsedTime" << std::endl;
    for (int i = 0; i < iNumberOfFixedParameters; ++i) {
        output << fixedParameters[i] << " ";
    }
    for (int i = 0; i<  iNumberOfParameters; ++i) {
        output << parameters[i] << " ";
    }
    std::cout << "Registration parameters successfully written to file " << outfile << std::endl;

}

void MyITKImageHelper::writeTransform(itk::Euler3DTransform< PixelType >::ConstPointer transform,
    std::string outfile){

    const unsigned int dim = 3;

    itk::Euler3DTransform< PixelType >::ParametersType parameters = transform->GetParameters();
    itk::Euler3DTransform< PixelType >::ParametersType fixedParameters = transform->GetFixedParameters();

    const unsigned int iNumberOfParameters = parameters.size();
    const unsigned int iNumberOfFixedParameters = fixedParameters.size();

    std::ofstream output(outfile);

    if(!output.is_open()){
        throw MyException("Cannot open the file to write");
    }

    // Define output precision
    output.precision(10); 

    // output << "iteration \t" << "elapsedTime" << std::endl;
    for (int i = 0; i < iNumberOfFixedParameters; ++i) {
        output << fixedParameters[i] << " ";
    }
    for (int i = 0; i<  iNumberOfParameters; ++i) {
        output << parameters[i] << " ";
    }
    std::cout << "Registration parameters successfully written to file " << outfile << std::endl;
}

void MyITKImageHelper::writeTransform(itk::ScaledTranslationEuler3DTransform< PixelType >::ConstPointer transform,
    std::string outfile){

    const unsigned int dim = 3;

    itk::ScaledTranslationEuler3DTransform< PixelType >::ParametersType parameters = transform->GetParameters();
    itk::ScaledTranslationEuler3DTransform< PixelType >::ParametersType fixedParameters = transform->GetFixedParameters();

    const unsigned int iNumberOfParameters = parameters.size();
    const unsigned int iNumberOfFixedParameters = fixedParameters.size();

    std::ofstream output(outfile);

    if(!output.is_open()){
        throw MyException("Cannot open the file to write");
    }

    // Define output precision
    output.precision(10); 

    // output << "iteration \t" << "elapsedTime" << std::endl;
    for (int i = 0; i < iNumberOfFixedParameters; ++i) {
        output << fixedParameters[i] << " ";
    }
    for (int i = 0; i<  iNumberOfParameters; ++i) {
        output << parameters[i] << " ";
    }
    std::cout << "Registration parameters successfully written to file " << outfile << std::endl;
}

