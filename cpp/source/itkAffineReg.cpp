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

// My includes
#include "MyITKImageHelper.h"

// Global variables

// Typedefs

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

  const unsigned int Dimension = 3;

  //  The transform that will map the fixed image into the moving image.
  typedef itk::AffineTransform< PixelType, Dimension > TransformType;
  
  //  An optimizer is required to explore the parameter space of the transform
  //  in search of optimal values of the metric.
  typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
  
  //  The metric will compare how well the two images match each other. Metric
  //  types are usually parameterized by the image types as it can be seen in
  //  the following type declaration.
  typedef itk::MeanSquaresImageToImageMetric< ImageType3D, ImageType3D > MetricType;
  
  //  Finally, the type of the interpolator is declared. The interpolator will
  //  evaluate the intensities of the moving image at non-grid positions.
  typedef itk:: LinearInterpolateImageFunction< ImageType3D, PixelType > InterpolatorType;
  
  //  The registration method type is instantiated using the types of the
  //  fixed and moving images. This class is responsible for interconnecting
  //  all the components that we have described so far.
  typedef itk::ImageRegistrationMethod< ImageType3D, ImageType3D > RegistrationType;

  // Create components
  MetricType::Pointer         metric        = MetricType::New();
  TransformType::Pointer      transform     = TransformType::New();
  OptimizerType::Pointer      optimizer     = OptimizerType::New();
  InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
  RegistrationType::Pointer   registration  = RegistrationType::New();
  
  // Each component is now connected to the instance of the registration method.
  registration->SetMetric(        metric        );
  registration->SetOptimizer(     optimizer     );
  registration->SetTransform(     transform     );
  registration->SetInterpolator(  interpolator  );

  return EXIT_SUCCESS;
}
 