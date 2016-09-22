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

  MyITKImageHelper::executeShellCommand(cmd);
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

  MyITKImageHelper::executeShellCommand(cmd);
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

  MyITKImageHelper::executeShellCommand(cmd);
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

  MyITKImageHelper::executeShellCommand(cmd);
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

  MyITKImageHelper::executeShellCommand(cmd);
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

  MyITKImageHelper::executeShellCommand(cmd);
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

  MyITKImageHelper::executeShellCommand(cmd);
}

// show 3D image + 2 overlays
void MyITKImageHelper::showImage(const ImageType3D::Pointer image, const ImageType3D::Pointer overlay, const ImageType3D::Pointer overlay2, const std::string &filename){

  const std::string dir_output = "/tmp/";

  // Write image
  MyITKImageHelper::writeImage(image, dir_output + filename + ".nii.gz");
  MyITKImageHelper::writeImage(overlay, dir_output + filename + "_overlay.nii.gz");
  MyITKImageHelper::writeImage(overlay2, dir_output + filename + "_overlay2.nii.gz");

  // View image with overlay via itksnap
  std::string  cmd = "itksnap ";
  cmd += "-g " + dir_output + filename + ".nii.gz ";
  cmd += "-o " + dir_output + filename + "_overlay.nii.gz ";
  cmd += " " + dir_output + filename + "_overlay2.nii.gz ";
  cmd += "& ";

  // image with its segmentation
  // std::string  cmd = "itksnap ";
  // cmd += "-g " + dir_output + filename + ".nii.gz ";
  // cmd += "-s " + dir_output + filename + "_segmentation.nii.gz ";
  // cmd += "& ";

  MyITKImageHelper::executeShellCommand(cmd);
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

  MyITKImageHelper::executeShellCommand(cmd);
}

void MyITKImageHelper::showImage(const std::vector<ImageType3D::Pointer>& images, const std::string &filename){
    
    std::vector<std::string> filenames;
    for(std::vector<ImageType3D::Pointer>::size_type i = 0; i < images.size(); ++i){
        filenames.push_back(filename + "_" + std::to_string(i));
    }
    MyITKImageHelper::showImage(images, filenames);
}

void MyITKImageHelper::showImage(const std::vector<ImageType3D::Pointer> &images, const std::vector<std::string> &filenames){

    const std::string dir_output = "/tmp/";

    // View images via itksnap
    std::string  cmd = "itksnap ";
    cmd += "-g " + dir_output + filenames[0] + ".nii.gz ";
    cmd += "-o ";

    MyITKImageHelper::writeImage(images[0], dir_output + filenames[0] + ".nii.gz");

    for(std::vector<ImageType3D::Pointer>::size_type i = 1; i < images.size(); ++i) {
        MyITKImageHelper::writeImage(images[i], dir_output + filenames[i] + ".nii.gz");
        cmd += dir_output + filenames[i] + ".nii.gz ";
    }
    cmd += "& ";

    MyITKImageHelper::executeShellCommand(cmd);
}

void MyITKImageHelper::showImage(const std::vector<ImageType3D::Pointer> &images, const std::string *filenames){

    const std::string dir_output = "/tmp/";

    // View images via itksnap
    std::string  cmd = "itksnap ";
    cmd += "-g " + dir_output + filenames[0] + ".nii.gz ";
    cmd += "-o ";

    MyITKImageHelper::writeImage(images[0], dir_output + filenames[0] + ".nii.gz");

    for(std::vector<ImageType3D::Pointer>::size_type i = 1; i < images.size(); ++i) {
        MyITKImageHelper::writeImage(images[i], dir_output + filenames[i] + ".nii.gz");
        cmd += dir_output + filenames[i] + ".nii.gz ";
    }
    cmd += "& ";

    MyITKImageHelper::executeShellCommand(cmd);
}

void MyITKImageHelper::showImage(const JacobianBaseType3D::Pointer dimage, const std::string &filename){

    const std::string dir_output = "/tmp/";

    // Use Index Selection Filter to select specific image corresponding to particular index
    MyITKImageHelper::IndexSelectionType3D::Pointer indexSelectionFilter3D = MyITKImageHelper::IndexSelectionType3D::New();
    indexSelectionFilter3D->SetInput( dimage );

    // Write each image w.r.t. index
    for (int i = 0; i < 3; ++i)
    {
        indexSelectionFilter3D->SetIndex(i);
        indexSelectionFilter3D->Update();
        MyITKImageHelper::writeImage(indexSelectionFilter3D->GetOutput(), dir_output + filename + "_" + std::to_string(i) + ".nii.gz");
    }

    // View images via itksnap
    std::string  cmd = "itksnap ";
    cmd += "-g " + dir_output + filename + "_0.nii.gz ";
    cmd += "-o ";
    for (int i = 1; i < 3; ++i)
    {
        cmd += dir_output + filename + "_" + std::to_string(i) + ".nii.gz ";
    }
    cmd += "& ";

    MyITKImageHelper::executeShellCommand(cmd);
}

/** Write image */
void MyITKImageHelper::writeImage(const ImageType2D::Pointer image, const std::string &filename, const bool bVerbose){

  // Create reader for nifti images
  itk::ImageFileWriter< ImageType2D >::Pointer writer = itk::ImageFileWriter< ImageType2D >::New();
  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
  writer->SetImageIO(imageIO);

  // Write images
  writer->SetFileName( filename );
  writer->SetInput( image );
  writer->Update();

  if (bVerbose){
    std::cout << "Image successfully written to file " << filename << std::endl;
  }
}
void MyITKImageHelper::writeImage(const MaskImageType2D::Pointer image, const std::string &filename, const bool bVerbose){

  // Create reader for nifti images
  itk::ImageFileWriter< MaskImageType2D >::Pointer writer = itk::ImageFileWriter< MaskImageType2D >::New();
  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
  writer->SetImageIO(imageIO);

  // Write images
  writer->SetFileName( filename );
  writer->SetInput( image );
  writer->Update();

  if (bVerbose){
    std::cout << "Image successfully written to file " << filename << std::endl;
  }
}
void MyITKImageHelper::writeImage(const ImageType3D::Pointer image, const std::string &filename, const bool bVerbose){

  // Create reader for nifti images
  itk::ImageFileWriter< ImageType3D >::Pointer writer = itk::ImageFileWriter< ImageType3D >::New();
  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
  writer->SetImageIO(imageIO);

  // Write images 
  writer->SetFileName( filename );
  writer->SetInput( image );
  writer->Update();

  if (bVerbose){
    std::cout << "Image successfully written to file " << filename << std::endl;
  }
}
void MyITKImageHelper::writeImage(const MaskImageType3D::Pointer image, const std::string &filename, const bool bVerbose){

  // Create reader for nifti images
  itk::ImageFileWriter< MaskImageType3D >::Pointer writer = itk::ImageFileWriter< MaskImageType3D >::New();
  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
  writer->SetImageIO(imageIO);

  // Write images 
  writer->SetFileName( filename );
  writer->SetInput( image );
  writer->Update();

  if (bVerbose){
    std::cout << "Image successfully written to file " << filename << std::endl;
  }
}

void MyITKImageHelper::printTransform(itk::AffineTransform< PixelType, 3 >::ConstPointer transform){
  
  const unsigned int dim = 3;

  itk::Matrix< PixelType, dim, dim >  matrix = transform->GetMatrix();

  itk::AffineTransform< PixelType, dim >::ParametersType parameters = transform->GetParameters();
  itk::AffineTransform< PixelType, dim >::ParametersType center = transform->GetFixedParameters();

  std::cout << "AffineTransform:" << std::endl;

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

  std::cout << "Euler3DTransform:" << std::endl;

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

  std::cout << "ScaledTranslationEuler3DTransform:" << std::endl;

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

void MyITKImageHelper::printTransform(itk::InplaneSimilarity3DTransform< PixelType >::ConstPointer transform){
  
  const unsigned int dim = 3;

  itk::Matrix< PixelType, dim, dim >  matrix = transform->GetMatrix();

  itk::InplaneSimilarity3DTransform< PixelType >::ParametersType parameters = transform->GetParameters();
  itk::InplaneSimilarity3DTransform< PixelType >::ParametersType center = transform->GetFixedParameters();

  std::cout << "InplaneSimilarity3DTransform:" << std::endl;

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

  std::cout << "\t scale (in-plane) = " << std::endl;
  printf("\t\t%.4f\n", transform->GetScale());

}

void MyITKImageHelper::writeTransform(itk::AffineTransform< PixelType, 3 >::ConstPointer transform,
    std::string outfile, const bool bVerbose){

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
    if (bVerbose){
      std::cout << "Registration parameters successfully written to file " << outfile << std::endl;
    }

}

void MyITKImageHelper::writeTransform(itk::Euler3DTransform< PixelType >::ConstPointer transform,
    std::string outfile, const bool bVerbose){

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
    if (bVerbose){
      std::cout << "Registration parameters successfully written to file " << outfile << std::endl;
    }
}

void MyITKImageHelper::writeTransform(itk::ScaledTranslationEuler3DTransform< PixelType >::ConstPointer transform,
    std::string outfile, const bool bVerbose){

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
    if (bVerbose){
      std::cout << "Registration parameters successfully written to file " << outfile << std::endl;
    }
}

void MyITKImageHelper::writeTransform(itk::InplaneSimilarity3DTransform< PixelType >::ConstPointer transform,
    std::string outfile, const bool bVerbose){

    const unsigned int dim = 3;

    itk::InplaneSimilarity3DTransform< PixelType >::ParametersType parameters = transform->GetParameters();
    itk::InplaneSimilarity3DTransform< PixelType >::ParametersType fixedParameters = transform->GetFixedParameters();

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
    if (bVerbose){
      std::cout << "Registration parameters successfully written to file " << outfile << std::endl;
    }
}

void MyITKImageHelper::executeShellCommand(const std::string &cmd){
    std::cout << cmd << std::endl;
    system(cmd.c_str());
}
