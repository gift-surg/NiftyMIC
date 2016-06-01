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
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <itkImageRegistrationMethodv4.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>

#include <itkMeanSquaresImageToImageMetricv4.h>
#include <itkMattesMutualInformationImageToImageMetricv4.h>
#include <itkCorrelationImageToImageMetricv4.h>

#include <itkRegularStepGradientDescentOptimizerv4.h>
#include <itkLBFGSOptimizerv4.h>

#include <itkResampleImageFilter.h>
// #include <itkRescaleIntensityImageFilter.h>

#include <itkAffineTransform.h>
#include <itkEuler3DTransform.h>
#include <itkImageMaskSpatialObject.h>

#include <itkRegistrationParameterScalesFromJacobian.h>
#include <itkRegistrationParameterScalesFromIndexShift.h>
#include <itkRegistrationParameterScalesFromPhysicalShift.h>

// My includes
#include "MyITKImageHelper.h"
#include "itkOrientedGaussianInterpolateImageFunction.h"
#include "readCommandLine.h"
#include "MyException.h"

// Global variables
const unsigned int Dimension = 3;

// Typedefs 
typedef itk::AffineTransform< PixelType, Dimension > TransformType;
// typedef itk::Euler3DTransform< PixelType > TransformType;

typedef itk::RegularStepGradientDescentOptimizerv4< PixelType > OptimizerType;
// typedef itk::LBFGSOptimizerv4 OptimizerType;

// typedef itk::MeanSquaresImageToImageMetricv4< ImageType3D, ImageType3D > MetricType;
// typedef itk::CorrelationImageToImageMetricv4< ImageType3D, ImageType3D > MetricType;
typedef itk::MattesMutualInformationImageToImageMetricv4< ImageType3D, ImageType3D > MetricType;

typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType> ScalesEstimatorType;
// typedef itk::RegistrationParameterScalesFromIndexShift<MetricType> ScalesEstimatorType;
// typedef itk::RegistrationParameterScalesFromJacobian<MetricType> ScalesEstimatorType;

typedef itk::LinearInterpolateImageFunction< ImageType3D, PixelType > InterpolatorType;
typedef itk::OrientedGaussianInterpolateImageFunction< ImageType3D, PixelType >  OrientedGaussianInterpolatorType;

typedef itk::ImageRegistrationMethodv4< ImageType3D, ImageType3D, TransformType > RegistrationType;

typedef itk::ResampleImageFilter< ImageType3D, ImageType3D > ResampleFilterType;
typedef itk::ResampleImageFilter< MaskImageType3D, MaskImageType3D > MaskResampleFilterType;

typedef itk::ImageMaskSpatialObject< Dimension > MaskType;

int main(int argc, char** argv)
{

  try{

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

      const bool bUseFixedMask = true;
      const bool bUseMovingMask = true;

      */

    //***Parse input of command line
    std::vector<std::string> input = readCommandLine(argc, argv);

    //***Check for empty vector ==> It was given "--help" in command line
    if( input[0] == "help request" ){
        return EXIT_SUCCESS;
    }

    //***Read input data of command line
    const std::string sFixed = input[0];
    const std::string sMoving = input[1];
    const std::string sFixedMask = input[2];
    const std::string sMovingMask = input[3];

    bool bUseMovingMask = false;
    bool bUseFixedMask = false;

    if(!sFixedMask.empty()){
        bUseFixedMask = true;
    }
    if(!sMovingMask.empty()){
        bUseMovingMask = true;
    }

    // Read images
    const ImageType3D::Pointer moving = MyITKImageHelper::readImage<ImageType3D>(sMoving + ".nii.gz");
    const ImageType3D::Pointer fixed = MyITKImageHelper::readImage<ImageType3D>(sFixed + ".nii.gz");

    // Read masks (if given)
    MaskImageType3D::Pointer fixedMask;
    MaskImageType3D::Pointer movingMask;
    
    if(bUseFixedMask){
        fixedMask = MyITKImageHelper::readImage<MaskImageType3D>(sFixedMask + ".nii.gz");
    }
    if(bUseMovingMask){
        movingMask = MyITKImageHelper::readImage<MaskImageType3D>(sMovingMask + ".nii.gz");
    }


    // MyITKImageHelper::showImage(moving, "moving");
    // MyITKImageHelper::showImage(fixed, fixedMask, "fixed");

    // Create components
    const MetricType::Pointer         metric                  = MetricType::New();
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

    // registration->SetInterpolator(  interpolator  );
    const double alpha = 2;
    itk::Vector<double, 9> covariance;
    covariance.Fill(0);
    covariance[0] = 0.26786367;
    covariance[4] = 0.26786367;
    covariance[8] = 2.67304559;

    interpolatorOrientedGaussian->SetCovariance( covariance );
    interpolatorOrientedGaussian->SetAlpha( alpha );
    // metric->SetFixedInterpolator(  interpolatorOrientedGaussian  );
    metric->SetMovingInterpolator(  interpolatorOrientedGaussian  );
    // metric->SetFixedInterpolator(  interpolator  );
    // metric->SetMovingInterpolator(  interpolator  );

    // Set Masks
    if(bUseFixedMask){
        spatialObjectFixedMask->SetImage( fixedMask );
        metric->SetFixedImageMask( spatialObjectFixedMask );
    }
    if(bUseMovingMask){
        spatialObjectMovingMask->SetImage( movingMask );
        metric->SetMovingImageMask( spatialObjectMovingMask );
    }

    // Set the registration inputs
    registration->SetFixedImage(fixed);
    registration->SetMovingImage(moving);
    // registration->SetFixedImageRegion( fixed->GetLargestPossibleRegion() );
   
    //  Initialize the transform
    TransformType::Pointer   identityTransform = TransformType::New();
    identityTransform->SetIdentity();
    registration->SetFixedInitialTransform( identityTransform );

    // Scales estimator
    ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric( metric );
    scalesEstimator->SetTransformForward( true );
    // scalesEstimator->SetSmallParameterVariation( 1.0 );

    // Multi-resolution framework
    const unsigned int numberOfLevels = 3;
    RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize( numberOfLevels );
    shrinkFactorsPerLevel[0] = 4;
    shrinkFactorsPerLevel[1] = 2;
    shrinkFactorsPerLevel[2] = 1;

    RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize( numberOfLevels );
    smoothingSigmasPerLevel[0] = 2;
    smoothingSigmasPerLevel[1] = 1;
    smoothingSigmasPerLevel[2] = 0;

    registration->SetNumberOfLevels ( numberOfLevels );
    registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
    registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );

    // Parametrize optimizer
    optimizer->SetDoEstimateLearningRateOnce( true );
    optimizer->SetMinimumStepLength( 0.01 );
    // optimizer->SetMaximumStepLength( 0.1 ); // If this is set too high, you will get a
    //"itk::ERROR: MeanSquaresImageToImageMetric(0xa27ce70): Too many samples map outside moving image buffer: 1818 / 10000" error
    optimizer->SetNumberOfIterations( 200 );

    // For LBFGS Optimizer
    // optimizer->SetDefaultStepLength( 1.5 );
    // optimizer->SetGradientConvergenceTolerance( 5e-2 );
    // optimizer->SetLineSearchAccuracy( 1.2 );
    // optimizer->TraceOn();
    // optimizer->SetMaximumNumberOfFunctionEvaluations( 1000 );
    optimizer->SetScalesEstimator( scalesEstimator );
    optimizer->SetMinimumConvergenceValue( 1e-6 );


    try {
      registration->Update();
      std::cout << "Optimizer stop condition: "
      << registration->GetOptimizer()->GetStopConditionDescription()
      << std::endl;
    }
    catch( itk::ExceptionObject & err ) {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }

    //  The result of the registration process is an array of parameters that
    //  defines the spatial transformation in an unique way. This final result is
    //  obtained using the \code{GetLastTransformParameters()} method.
    TransformType::ConstPointer transform = registration->GetTransform();
   
    TransformType::ParametersType finalParameters = transform->GetParameters();
    std::cout << "Final parameters:" << std::endl;
    std::cout << "\t matrix = " << std::endl;
    for (int i = 0; i < 3; ++i) {
      printf("\t\t%.4f\t%.4f\t%.4f\n", finalParameters[i*3], finalParameters[i*3+1], finalParameters[i*3+2]);
    }
    std::cout << "\t translation = " << std::endl;
    printf("\t\t%.4f\t%.4f\t%.4f\n", finalParameters[9], finalParameters[9+1], finalParameters[9+2]);
   
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

    // MyITKImageHelper::showImage(fixed, movingWarped, "fixed_moving");
    // MyITKImageHelper::showImage(movingWarped, movingMaskWarped, "fixed_mask");

  }
  catch(std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
        // std::cout << "EXIT_FAILURE = " << EXIT_FAILURE << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
 