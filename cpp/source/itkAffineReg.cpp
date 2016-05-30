/*! \brief
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date May 2016
 */

#include <iostream>
#include <string>
#include <limits.h>     /* PATH_MAX */
#include <math.h>
#include <cstdlib>     /* system, NULL, EXIT_FAILURE */
#include <chrono>

#include <itkImage.h>
#include <itkImageRegistrationMethod.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMeanSquaresImageToImageMetric.h>
#include <itkMattesMutualInformationImageToImageMetric.h>
#include <itkNormalizedCorrelationImageToImageMetric.h>
#include <itkRegularStepGradientDescentOptimizer.h>
#include <itkResampleImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkAffineTransform.h>
#include <itkEuler3DTransform.h>
#include <itkImageMaskSpatialObject.h>

// My includes
#include "MyITKImageHelper.h"
#include "itkOrientedGaussianInterpolateImageFunction.h"
#include "readCommandLine.h"
#include "MyException.h"

// Global variables
const unsigned int Dimension = 3;

// Typedefs 
typedef itk::AffineTransform< PixelType, Dimension > TransformType;
typedef itk::RegularStepGradientDescentOptimizer OptimizerType;

// typedef itk::MeanSquaresImageToImageMetric< ImageType3D, ImageType3D > MetricType;
// typedef itk::MattesMutualInformationImageToImageMetric< ImageType3D, ImageType3D > MetricType;
typedef itk::NormalizedCorrelationImageToImageMetric< ImageType3D, ImageType3D > MetricType;


typedef itk::LinearInterpolateImageFunction< ImageType3D, PixelType > InterpolatorType;
typedef itk::OrientedGaussianInterpolateImageFunction< ImageType3D, PixelType >  OrientedGaussianInterpolatorType;

typedef itk::ImageRegistrationMethod< ImageType3D, ImageType3D > RegistrationType;

typedef itk::ResampleImageFilter< ImageType3D, ImageType3D > ResampleFilterType;
typedef itk::ResampleImageFilter< MaskImageType3D, MaskImageType3D > MaskResampleFilterType;

typedef itk::ImageMaskSpatialObject< Dimension > MaskType;

int main(int argc, char** argv)
{
  /*
  // Define input
  const std::string dir_input = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/data/test/";

  // const std::string dir_output = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/cpp/ITK_Examples/MyFunctions/results/";

  // const std::string filename_image_2D = "2D_SheppLoganPhantom_512";
  // const std::string filename_image_3D = "3D_SingleDot_50";
  // const std::string filename_image_3D = "3D_Cross_50";
  // const std::string filename_image_3D = "3D_SheppLoganPhantom_64";
  // const std::string filename_image_3D = "fetal_brain_c";
  // const std::string filename_image_3D = "HR_volume_postmortem";
  const std::string sMoving = "fetal_brain_0";
  const std::string sFixed = "fetal_brain_1";

  const std::string filename_moving_mask = sMoving + "_mask";
  const std::string filename_fixed_mask = sFixed + "_mask";

  // Define output  
  // const std::string dir_output = "/tmp/";
  const std::string dir_output = "../../results/";
  const std::string filename_output = "test_output";

  // Read images
  const ImageType3D::Pointer moving = MyITKImageHelper::readImage<ImageType3D>(dir_input + sMoving + ".nii.gz");
  const ImageType3D::Pointer fixed = MyITKImageHelper::readImage<ImageType3D>(dir_input + sFixed + ".nii.gz");

  const MaskImageType3D::Pointer movingMask = MyITKImageHelper::readImage<MaskImageType3D>(dir_input + filename_moving_mask + ".nii.gz");
  const MaskImageType3D::Pointer fixedMask = MyITKImageHelper::readImage<MaskImageType3D>(dir_input + filename_fixed_mask + ".nii.gz");
  */

  try{
    //***Parse input of command line
    std::vector<std::string> input = readCommandLine(argc, argv);

    //***Check for empty vector ==> It was given "--help" in command line
    if( input[0] == "help request" ){
        return EXIT_SUCCESS;
    }

    //***Read input data of command line
    std::string sFixed = input[0];
    std::string sMoving = input[1];

    // Read images
    const ImageType3D::Pointer moving = MyITKImageHelper::readImage<ImageType3D>(sMoving + ".nii.gz");
    const ImageType3D::Pointer fixed = MyITKImageHelper::readImage<ImageType3D>(sFixed + ".nii.gz");


    MyITKImageHelper::showImage(moving, "moving");
    // MyITKImageHelper::showImage(fixed, fixedMask, "fixed");

    // Create components
    const MetricType::Pointer         metric                  = MetricType::New();
    const TransformType::Pointer      transform               = TransformType::New();
    const OptimizerType::Pointer      optimizer               = OptimizerType::New();
    const InterpolatorType::Pointer   interpolator            = InterpolatorType::New();
    const OrientedGaussianInterpolatorType::Pointer   interpolatorOrientedGaussian  = OrientedGaussianInterpolatorType::New();
    const RegistrationType::Pointer   registration            = RegistrationType::New();
    const MaskType::Pointer           spatialObjectFixedMask  = MaskType::New();
    const MaskType::Pointer           spatialObjectMovingMask = MaskType::New();
    const ResampleFilterType::Pointer resampler               = ResampleFilterType::New();
    const MaskResampleFilterType::Pointer resamplerMask       = MaskResampleFilterType::New();
    
    // Each component is now connected to the instance of the registration method.
    registration->SetMetric(        metric        );
    registration->SetOptimizer(     optimizer     );
    registration->SetTransform(     transform     );

    // registration->SetInterpolator(  interpolator  );
    const double alpha = 2;
    itk::Vector<double, 9> covariance;
    covariance.Fill(0);
    covariance[0] = 0.26786367;
    covariance[4] = 0.26786367;
    covariance[8] = 2.67304559;

    interpolatorOrientedGaussian->SetCovariance( covariance );
    interpolatorOrientedGaussian->SetAlpha( alpha );
    registration->SetInterpolator(  interpolatorOrientedGaussian  );

    

    // Set Masks
    // spatialObjectMovingMask->SetImage(movingMask);
    // spatialObjectFixedMask->SetImage(fixedMask);
    // metric->SetFixedImageMask( spatialObjectFixedMask );
    // metric->SetMovingImageMask( spatialObjectMovingMask );

    // Set the registration inputs
    registration->SetFixedImage(fixed);
    registration->SetMovingImage(moving);
   
    registration->SetFixedImageRegion( fixed->GetLargestPossibleRegion() );
   
    //  Initialize the transform
    typedef RegistrationType::ParametersType ParametersType;
    ParametersType initialParameters( transform->GetNumberOfParameters() );

    // rotation matrix
    initialParameters[0] = 1.0;  // R(0,0)
    initialParameters[1] = 0.0;  // R(0,1)
    initialParameters[2] = 0.0;  // R(0,2)
    initialParameters[3] = 0.0;  // R(1,0)
    initialParameters[4] = 1.0;  // R(1,1)
    initialParameters[5] = 0.0;  // R(1,2)
    initialParameters[6] = 0.0;  // R(2,0)
    initialParameters[7] = 0.0;  // R(2,1)
    initialParameters[8] = 1.0;  // R(2,2)
   
    // translation vector
    initialParameters[9]  = 0.0;
    initialParameters[10] = 0.0;
    initialParameters[11] = 0.0;

    registration->SetInitialTransformParameters( initialParameters );

    optimizer->SetMaximumStepLength( 0.1 ); // If this is set too high, you will get a
    //"itk::ERROR: MeanSquaresImageToImageMetric(0xa27ce70): Too many samples map outside moving image buffer: 1818 / 10000" error
   
    optimizer->SetMinimumStepLength( 0.01 );
   
    // Set a stopping criterion
    optimizer->SetNumberOfIterations( 200 );

    // Connect an observer
    //CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    //optimizer->AddObserver( itk::IterationEvent(), observer );

    try {
      registration->Update();
    }
    catch( itk::ExceptionObject & err ) {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }

    //  The result of the registration process is an array of parameters that
    //  defines the spatial transformation in an unique way. This final result is
    //  obtained using the \code{GetLastTransformParameters()} method.
   
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
    resampler->SetInterpolator( interpolator );
    resampler->Update();

    // resamplerMask->SetOutputParametersFromImage( fixedMask );
    // resamplerMask->SetInput( movingMask );
    // resamplerMask->SetTransform( registration->GetOutput()->Get() );
    // resamplerMask->SetDefaultPixelValue( 0.0 );
    // resamplerMask->Update();

    const ImageType3D::Pointer movingWarped = resampler->GetOutput();
    movingWarped->DisconnectPipeline();

    const MaskImageType3D::Pointer movingMaskWarped = resamplerMask->GetOutput();
    movingMaskWarped->DisconnectPipeline();

    MyITKImageHelper::showImage(fixed, movingWarped, "fixed_moving");
    // MyITKImageHelper::showImage(movingWarped, movingMaskWarped, "fixed_mask");

  }
  catch(std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
        // std::cout << "EXIT_FAILURE = " << EXIT_FAILURE << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
 