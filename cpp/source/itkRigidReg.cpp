/*! \brief Code
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
#include <itkImageRegistrationMethod.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMeanSquaresImageToImageMetric.h>
#include <itkMattesMutualInformationImageToImageMetric.h>
#include <itkNormalizedCorrelationImageToImageMetric.h>
#include <itkRegularStepGradientDescentOptimizer.h>
#include <itkResampleImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkEuler3DTransform.h>
#include <itkImageMaskSpatialObject.h>

// My includes
#include "MyITKImageHelper.h"

// Global variables
const unsigned int Dimension = 3;

// Typedefs 
typedef itk::Euler3DTransform< PixelType > TransformType;
typedef itk::RegularStepGradientDescentOptimizer OptimizerType;

// typedef itk::MeanSquaresImageToImageMetric< ImageType3D, ImageType3D > MetricType;
// typedef itk::MattesMutualInformationImageToImageMetric< ImageType3D, ImageType3D > MetricType;
typedef itk::NormalizedCorrelationImageToImageMetric< ImageType3D, ImageType3D > MetricType;


typedef itk:: LinearInterpolateImageFunction< ImageType3D, PixelType > InterpolatorType;

typedef itk::ImageRegistrationMethod< ImageType3D, ImageType3D > RegistrationType;

typedef itk::ResampleImageFilter< ImageType3D, ImageType3D > ResampleFilterType;
typedef itk::ResampleImageFilter< MaskImageType3D, MaskImageType3D > MaskResampleFilterType;

typedef itk::ImageMaskSpatialObject< Dimension > MaskType;

int main(int, char*[])
{

  // Define input
  const std::string dir_input = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/data/test/";

  // const std::string dir_output = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/cpp/ITK_Examples/MyFunctions/results/";

  // const std::string filename_image_2D = "2D_SheppLoganPhantom_512";
  // const std::string filename_image_3D = "3D_SingleDot_50";
  // const std::string filename_image_3D = "3D_Cross_50";
  // const std::string filename_image_3D = "3D_SheppLoganPhantom_64";
  // const std::string filename_image_3D = "fetal_brain_c";
  // const std::string filename_image_3D = "HR_volume_postmortem";
  const std::string filename_moving = "fetal_brain_0";
  const std::string filename_fixed = "fetal_brain_1";

  const std::string filename_moving_mask = filename_moving + "_mask";
  const std::string filename_fixed_mask = filename_fixed + "_mask";

  // Define output  
  // const std::string dir_output = "/tmp/";
  const std::string dir_output = "../../results/";
  const std::string filename_output = "test_output";

  // Read images
  const ImageType3D::Pointer moving = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_moving + ".nii.gz");
  const ImageType3D::Pointer fixed = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_fixed + ".nii.gz");

  const MaskImageType3D::Pointer movingMask = MyITKImageHelper::readImage<MaskImageType3D>(dir_input + filename_moving_mask + ".nii.gz");
  const MaskImageType3D::Pointer fixedMask = MyITKImageHelper::readImage<MaskImageType3D>(dir_input + filename_fixed_mask + ".nii.gz");
  
  // MyITKImageHelper::showImage(moving, movingMask, "moving");
  // MyITKImageHelper::showImage(fixed, fixedMask, "fixed");

  // Create components
  MetricType::Pointer         metric                  = MetricType::New();
  TransformType::Pointer      transform               = TransformType::New();
  OptimizerType::Pointer      optimizer               = OptimizerType::New();
  InterpolatorType::Pointer   interpolator            = InterpolatorType::New();
  RegistrationType::Pointer   registration            = RegistrationType::New();
  MaskType::Pointer           spatialObjectFixedMask  = MaskType::New();
  MaskType::Pointer           spatialObjectMovingMask = MaskType::New();
  ResampleFilterType::Pointer resampler               = ResampleFilterType::New();
  MaskResampleFilterType::Pointer resamplerMask       = MaskResampleFilterType::New();
  
  // Each component is now connected to the instance of the registration method.
  registration->SetMetric(        metric        );
  registration->SetOptimizer(     optimizer     );
  registration->SetTransform(     transform     );
  registration->SetInterpolator(  interpolator  );

  // Set Masks
  spatialObjectMovingMask->SetImage(movingMask);
  spatialObjectFixedMask->SetImage(fixedMask);
  metric->SetFixedImageMask( spatialObjectFixedMask );
  metric->SetMovingImageMask( spatialObjectMovingMask );

  // Set the registration inputs
  registration->SetFixedImage(fixed);
  registration->SetMovingImage(moving);
 
  registration->SetFixedImageRegion( fixed->GetLargestPossibleRegion() );
 
  //  Initialize the transform
  transform->SetIdentity();

  registration->SetInitialTransformParameters( transform->GetParameters() );

  optimizer->SetMaximumStepLength( 0.1 ); // If this is set too high, you will get a
  //"itk::ERROR: MeanSquaresImageToImageMetric(0xa27ce70): Too many samples map outside moving image buffer: 1818 / 10000" error
 
  optimizer->SetMinimumStepLength( 0.01 );
 
  // Set a stopping criterion
  optimizer->SetNumberOfIterations( 200 );

  // Connect an observer
  //CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  //optimizer->AddObserver( itk::IterationEvent(), observer );

  try
  {
    registration->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  //  The result of the registration process is an array of parameters that
  //  defines the spatial transformation in an unique way. This final result is
  //  obtained using the \code{GetLastTransformParameters()} method.
  typedef RegistrationType::ParametersType ParametersType;
  ParametersType finalParameters = registration->GetLastTransformParameters();
  std::cout << "Final parameters: " << finalParameters << std::endl;
 
  //  The value of the image metric corresponding to the last set of parameters
  //  can be obtained with the \code{GetValue()} method of the optimizer.
  const double bestValue = optimizer->GetValue();
 
  // Print out results
  std::cout << "Result = " << std::endl;
  std::cout << " Metric value  = " << bestValue          << std::endl;

  // Resample registered image
  resampler->SetOutputParametersFromImage( fixed );
  // resampler->SetSize( fixed->GetLargestPossibleRegion().GetSize() );
  // resampler->SetOutputOrigin(  fixed->GetOrigin() );
  // resampler->SetOutputSpacing( fixed->GetSpacing() );
  // resampler->SetOutputDirection( fixed->GetDirection() );
  resampler->SetInput( moving );
  resampler->SetTransform( registration->GetOutput()->Get() );
  resampler->SetDefaultPixelValue( 0.0 );
  resampler->Update();

  resamplerMask->SetOutputParametersFromImage( fixedMask );
  resamplerMask->SetInput( movingMask );
  resamplerMask->SetTransform( registration->GetOutput()->Get() );
  resamplerMask->SetDefaultPixelValue( 0.0 );
  resamplerMask->Update();

  const ImageType3D::Pointer movingWarped = resampler->GetOutput();
  movingWarped->DisconnectPipeline();

  const MaskImageType3D::Pointer movingMaskWarped = resamplerMask->GetOutput();
  movingMaskWarped->DisconnectPipeline();

  MyITKImageHelper::showImage(fixed, movingWarped, "fixed_moving");
  MyITKImageHelper::showImage(movingWarped, movingMaskWarped, "fixed_mask");


  return EXIT_SUCCESS;
}
 