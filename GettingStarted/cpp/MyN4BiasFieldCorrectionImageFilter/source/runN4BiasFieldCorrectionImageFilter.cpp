/*! \brief Code to play with itk::N4BiasFieldCorrectionImageFilter
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date May 2016
 */

#include <string>
#include <limits.h>     /* PATH_MAX */
#include <math.h>
#include <cstdlib>     /* system, NULL, EXIT_FAILURE */
#include <chrono>

#include <itkImage.h>
#include "itkN4BiasFieldCorrectionImageFilter.h"
#include "itkOtsuThresholdImageFilter.h"

// My includes
#include "MyITKImageHelper.h"

// Typedefs
typedef itk::OtsuThresholdImageFilter<ImageType3D, MaskImageType3D> ThresholderType3D;
typedef itk::N4BiasFieldCorrectionImageFilter<ImageType3D, MaskImageType3D, ImageType3D> CorrecterType3D;

int main(int, char*[])
{

  // Define input
  const std::string dir_input = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/data/test/";

  // const std::string dir_output = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/cpp/ITK_Examples/MyFunctions/results/";

  // const std::string filename_image_2D = "2D_SingleDot_50";
  // const std::string filename_image_2D = "2D_Cross_50";
  // const std::string filename_image_2D = "2D_Text";
  const std::string filename_image_2D = "2D_BrainWeb";
  // const std::string filename_image_2D = "2D_Lena_512";

  // const std::string filename_image_2D = "2D_SheppLoganPhantom_512";
  // const std::string filename_image_3D = "3D_SingleDot_50";
  // const std::string filename_image_3D = "3D_Cross_50";
  // const std::string filename_image_3D = "3D_SheppLoganPhantom_64";
  // const std::string filename_image_3D = "fetal_brain_c";
  // const std::string filename_image_3D = "HR_volume_postmortem";
  const std::string filename_image_3D = "fetal_brain_0";
  const std::string filename_image_3D_mask = filename_image_3D + "_mask";

  // Define output  
  // const std::string dir_output = "/tmp/";
  const std::string dir_output = "../../results/";
  const std::string filename_output = "test_output";

  // Read images
  const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D + ".nii.gz");
  const ImageType3D::Pointer image_3D = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_image_3D + ".nii.gz");
  MaskImageType3D::Pointer image_3D_mask = MyITKImageHelper::readImage<MaskImageType3D>(dir_input + filename_image_3D_mask + ".nii.gz");
  
  // MyITKImageHelper::showImage(image_2D);
  // MyITKImageHelper::showImage(image_3D, image_3D_mask);

  /*
        N4 Bias Field Correction Image Filter
  */
  // MaskImageType3D::Pointer image_3D_mask = ITK_NULLPTR;

  if( !image_3D_mask ){
    std::cout << "Mask not read.  Creating Otsu mask." << std::endl;
    ThresholderType3D::Pointer otsu = ThresholderType3D::New();
    
    otsu->SetInput( image_3D );
    // otsu->SetNumberOfHistogramBins( 200 );
    otsu->SetInsideValue( 0 );
    otsu->SetOutsideValue( 1 );

    otsu->Update();
    image_3D_mask = otsu->GetOutput();
    image_3D_mask->DisconnectPipeline();
  }
  MyITKImageHelper::showImage(image_3D, image_3D_mask);


  CorrecterType3D::Pointer correcter = CorrecterType3D::New();
  correcter->SetMaskLabel( 1 );
  correcter->SetSplineOrder( 3 );
  correcter->SetWienerFilterNoise( 0.01 );
  correcter->SetBiasFieldFullWidthAtHalfMaximum( 0.15 );
  correcter->SetConvergenceThreshold( 0.0000001 );

  // set the input image and mask image
  correcter->SetInput( image_3D );
  correcter->SetMaskImage( image_3D_mask );

  try{
    correcter->Update();
  }
  catch( itk::ExceptionObject &excep ){
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }

  const ImageType3D::Pointer image_3D_corrected = correcter->GetOutput();
  image_3D_corrected->DisconnectPipeline();

  MyITKImageHelper::showImage(image_3D, image_3D_corrected);
  MyITKImageHelper::writeImage(image_3D, dir_output + filename_image_3D + ".nii.gz");
  MyITKImageHelper::writeImage(image_3D_corrected, dir_output + filename_image_3D + "_corrected.nii.gz");

  return EXIT_SUCCESS;
}
 