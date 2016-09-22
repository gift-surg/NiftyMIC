/*! \brief Code to verify the implementation of itkAdjointOrientedGaussianInterpolateImageFilter.
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date February 2016
 */

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch.hpp>
#include <unistd.h>
#include <string>
#include <limits.h>     /* PATH_MAX */
#include <math.h>
#include <cstdlib>     /* system, NULL, EXIT_FAILURE */
// #include <iostream>
// #include <stdio.h>

#include <itkImage.h>
#include <itkResampleImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkAbsoluteValueDifferenceImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkEuler3DTransform.h>
#include <itkGradientImageFilter.h>
#include <itkGradientRecursiveGaussianImageFilter.h>
#include <itkDerivativeImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <itkSimilarity3DTransform.h>

#include <itkImageRegistrationMethodv4.h>
#include <itkCenteredTransformInitializer.h>

#include <itkMeanSquaresImageToImageMetricv4.h>
#include <itkMattesMutualInformationImageToImageMetricv4.h>
#include <itkCorrelationImageToImageMetricv4.h>

#include <itkInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>

#include <itkRegularStepGradientDescentOptimizerv4.h>
#include <itkRegistrationParameterScalesFromJacobian.h>
#include <itkRegistrationParameterScalesFromIndexShift.h>
#include <itkRegistrationParameterScalesFromPhysicalShift.h>

// My includes
#include "MyITKImageHelper.h"
#include "itkAdjointOrientedGaussianInterpolateImageFilter.h"
#include "itkOrientedGaussianInterpolateImageFilter.h"
#include "itkOrientedGaussianInterpolateImageFunction.h"
#include "itkGradientEuler3DTransformImageFilter.h"
#include "itkInplaneSimilarity3DTransform.h"
// #include "MyException.h"


// Typedefs
typedef itk::ResampleImageFilter< ImageType2D, ImageType2D >  FilterType_Resample_2D;
typedef itk::ResampleImageFilter< ImageType3D, ImageType3D >  FilterType_Resample_3D;

typedef itk::AdjointOrientedGaussianInterpolateImageFilter<ImageType2D,ImageType2D>  FilterType_AdjointOrientedGaussian_2D;
typedef itk::AdjointOrientedGaussianInterpolateImageFilter<ImageType3D,ImageType3D>  FilterType_AdjointOrientedGaussian_3D;

typedef itk::OrientedGaussianInterpolateImageFilter<ImageType2D,ImageType2D>  FilterType_OrientedGaussian_2D;
typedef itk::OrientedGaussianInterpolateImageFilter<ImageType3D,ImageType3D>  FilterType_OrientedGaussian_3D;

typedef itk::OrientedGaussianInterpolateImageFunction< ImageType2D, PixelType >  InterpolatorType_2D;
typedef itk::OrientedGaussianInterpolateImageFunction< ImageType3D, PixelType >  InterpolatorType_3D;

typedef itk::MultiplyImageFilter< ImageType2D, ImageType2D, ImageType2D> MultiplyImageFilter_2D;
typedef itk::MultiplyImageFilter< ImageType3D, ImageType3D, ImageType3D> MultiplyImageFilter_3D;

typedef itk::AddImageFilter< ImageType3D, ImageType3D, ImageType3D> AddImageFilter_3D;

typedef itk::AbsoluteValueDifferenceImageFilter< ImageType2D, ImageType2D, ImageType2D> AbsoluteValueDifferenceImageFilterType_2D;
typedef itk::AbsoluteValueDifferenceImageFilter< ImageType3D, ImageType3D, ImageType3D> AbsoluteValueDifferenceImageFilterType_3D;

typedef itk::StatisticsImageFilter<ImageType2D> StatisticsImageFilterType_2D;
typedef itk::StatisticsImageFilter<ImageType3D> StatisticsImageFilterType_3D;

// Transform Types
typedef itk::Euler3DTransform< PixelType > EulerTransformType;
typedef itk::InplaneSimilarity3DTransform< PixelType > InplaneSimilarityTransformType;
// typedef itk::Similarity3DTransform< PixelType > InplaneSimilarityTransformType;

// Optimizer Types
typedef itk::RegularStepGradientDescentOptimizerv4< PixelType > RegularStepGradientDescentOptimizerType;

// Interpolator Types
typedef itk::LinearInterpolateImageFunction< ImageType3D, PixelType > LinearInterpolatorType;
typedef itk::NearestNeighborInterpolateImageFunction< ImageType3D, PixelType > NearestNeighborInterpolatorType;

// Metric Types
typedef itk::MeanSquaresImageToImageMetricv4< ImageType3D, ImageType3D > MeanSquaresMetricType;
typedef itk::CorrelationImageToImageMetricv4< ImageType3D, ImageType3D > CorrelationMetricType;
typedef itk::MattesMutualInformationImageToImageMetricv4< ImageType3D, ImageType3D > MattesMutualInformationMetricType;

// Definitions used
typedef CorrelationMetricType MetricType;
// typedef MeanSquaresMetricType MetricType;
typedef InplaneSimilarityTransformType TransformType;
typedef LinearInterpolatorType InterpolatorType;
// typedef NearestNeighborInterpolatorType InterpolatorType;
typedef RegularStepGradientDescentOptimizerType OptimizerType;

typedef itk::ImageRegistrationMethodv4< ImageType3D, ImageType3D, TransformType > RegistrationType;
typedef itk::CenteredTransformInitializer< TransformType, ImageType3D, ImageType3D > TransformInitializerType;
// typedef itk::RegistrationParameterScalesFromJacobian< MetricType > ScalesEstimatorType;
typedef itk::RegistrationParameterScalesFromPhysicalShift< MetricType > ScalesEstimatorType;
// typedef itk::RegistrationParameterScalesFromIndexShift< MetricType > ScalesEstimatorType;

// Unit tests
TEST_CASE( "itkInplaneSimilarity3DTransform: Brain", 
  "[itkInplaneSimilarity3DTransform: Brain]") {


    const double tolerance = 1e-4;
    const double scale = 0.9;

    const unsigned int dimension = 3;

    // Define input and output
    const std::string dir_input = "../exampleData/";
    // const std::string filename = "3D_Cross_50.nii.gz";
    // const std::string filename = "3D_SheppLoganPhantom_64_rotated.nii.gz";
    const std::string filename = "FetalBrain_reconstruction_3stacks_myAlg.nii.gz";

    // Instantiate
    const RegistrationType::Pointer registration = RegistrationType::New();
    const MetricType::Pointer metric = MetricType::New();
    const InterpolatorType::Pointer interpolator = InterpolatorType::New();
    const OptimizerType::Pointer optimizer = OptimizerType::New();
    const ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    const FilterType_Resample_3D::Pointer resampler = FilterType_Resample_3D::New();
    TransformType::Pointer initialTransform = TransformType::New();

    // Read images
    const ImageType3D::Pointer fixed = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename);
    const ImageType3D::Pointer moving = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename);
    // MyITKImageHelper::showImage(fixed, "fixed");
    // MyITKImageHelper::showImage(moving, "moving");

    ImageType3D::DirectionType direction = fixed->GetDirection();
    // direction[0][0]=1;
    // direction[1][1]=1;
    // direction[2][2]=1;
    // direction[0][1]=0;
    // direction[0][2]=0;
    // direction[1][0]=0;
    // direction[1][2]=0;
    // direction[2][0]=0;
    // direction[2][1]=0;
    // fixed->SetDirection(direction);
    // moving->SetDirection(direction);

    // Generate test case: Alter image for registration
    ImageType3D::SpacingType spacing = moving->GetSpacing();
    spacing[0] *= scale;
    spacing[1] *= scale;
    moving->SetSpacing(spacing);

    TransformType::FixedParametersType fixedParameters = initialTransform->GetFixedParameters();
    TransformType::FixedParametersType fixedParameters_extended = initialTransform->GetFixedParameters();
    const unsigned int N_fixedParameters = fixedParameters.GetSize();
    fixedParameters_extended.SetSize(N_fixedParameters+9);

    // Copy previous fixed parameters
    for (int i = 0; i < N_fixedParameters; ++i)
    {
        fixedParameters_extended[i] = fixedParameters[i];
    }
    // Fill extended fixed parameters with direction information
    for (int i = 0; i < dimension; ++i)
    {
        for (int j = 0; j < dimension; ++j)
        {
            fixedParameters_extended[N_fixedParameters+dimension*i+j] = direction[i][j];
        }
    }
    initialTransform->SetFixedParameters(fixedParameters_extended);

    // Initialize the transform
    TransformInitializerType::Pointer initializer = TransformInitializerType::New();
    initializer->SetTransform(initialTransform);
    initializer->SetFixedImage( fixed );
    initializer->SetMovingImage( moving );
    initializer->GeometryOn();
    // initializer->MomentsOn();
    initializer->InitializeTransform();
    // initialTransform->Print(std::cout);

    registration->SetInitialTransform( initialTransform );
    registration->SetFixedInitialTransform( EulerTransformType::New() ); // Otherwise segmentation fault
    // registration->InPlaceOff();
    // registration->GetFixedInitialTransform()->Print(std::cout);
    // std::cout << "FixedParameters = " << registration->GetFixedInitialTransform()->GetFixedParameters() << std::endl;

    // Set metric
    metric->SetMovingInterpolator( interpolator );

    // Scales estimator
    scalesEstimator->SetMetric( metric );

    // Set optimizer
    optimizer->SetNumberOfIterations( 100 );
    // optimizer->SetGradientMagnitudeTolerance( 1e-6 ); //
    // optimizer->SetMinimumStepLength( 1e-6 );
    optimizer->SetScalesEstimator( scalesEstimator );
    // optimizer->SetDoEstimateLearningRateOnce( false );
    // optimizer->EstimateLearningRate();

    // Set registration
    registration->SetFixedImage(fixed);
    registration->SetMovingImage(moving);
    registration->SetMetric( metric );
    registration->SetOptimizer( optimizer );

    // Execute registration
    try {
      registration->Update();

      std::cout << "Optimizer stop condition: "
      << registration->GetOptimizer()->GetStopConditionDescription()
      << std::endl;
    }
    catch( itk::ExceptionObject & err ) {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      // return EXIT_FAILURE;
      throw MyException("ExeceptionObject caught during registration");
    }

    // Process registration results
    TransformType::ConstPointer transform = registration->GetTransform();    
    // transform->Print(std::cout);
    // std::cout << transform->GetParameters() << std::endl;
    // MyITKImageHelper::printTransform(transform);

    // Resample Image Filter
    resampler->SetOutputParametersFromImage(fixed);
    resampler->SetDefaultPixelValue( 0.0 );
    resampler->SetInterpolator( InterpolatorType::New() );
    resampler->SetInput(moving);
    resampler->Update();
    const ImageType3D::Pointer movingResampled = resampler->GetOutput();
    movingResampled->DisconnectPipeline();

    resampler->SetTransform( registration->GetOutput()->Get() );
    resampler->Update();

    const ImageType3D::Pointer movingWarped = resampler->GetOutput();
    movingWarped->DisconnectPipeline();

    std::vector<ImageType3D::Pointer> image_vector;
    image_vector.push_back(fixed);
    image_vector.push_back(movingResampled);
    image_vector.push_back(movingWarped);
    std::string titles_array[3] = {"array_fixed", "array_moving", "array_movingWarped"};
    MyITKImageHelper::showImage(image_vector, titles_array);


    // Filters to evaluate absolute difference
    // const StatisticsImageFilterType_3D::Pointer statisticsImageFilter_3D = StatisticsImageFilterType_3D::New();
    // const AbsoluteValueDifferenceImageFilterType_3D::Pointer absoluteValueDifferenceImageFilter_3D = AbsoluteValueDifferenceImageFilterType_3D::New();

    // absoluteValueDifferenceImageFilter_3D->SetInput1( fixed );
    // absoluteValueDifferenceImageFilter_3D->SetInput2( movingWarped );
    
    // statisticsImageFilter_3D->SetInput( absoluteValueDifferenceImageFilter_3D->GetOutput() );
    // statisticsImageFilter_3D->Update();
    // const double abs_diff = statisticsImageFilter_3D->GetSum();

    // Check accuracy
    const double scale_estimated = transform->GetScale();
    const double abs_diff = std::abs(scale-scale_estimated);
    std::cout << "|scale - scale_est|  = " << abs_diff << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}
