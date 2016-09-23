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
#include <itkGradientImageFilter.h>
#include <itkGradientRecursiveGaussianImageFilter.h>
#include <itkDerivativeImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <itkSimilarity3DTransform.h>

#include <itkEuler3DTransform.h>
#include <itkAffineTransform.h>

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
typedef itk::AffineTransform< PixelType, 3 > AffineTransformType;
// typedef itk::InplaneSimilarity3DTransform< PixelType > InplaneSimilarityTransformType;
typedef itk::Similarity3DTransform< PixelType > InplaneSimilarityTransformType;

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
    const AbsoluteValueDifferenceImageFilterType_3D::Pointer absoluteDifferenceImageFilter_3D = AbsoluteValueDifferenceImageFilterType_3D::New();
    const StatisticsImageFilterType_3D::Pointer statisticsImageFilter_3D = StatisticsImageFilterType_3D::New();
    const RegistrationType::Pointer registration = RegistrationType::New();
    const MetricType::Pointer metric = MetricType::New();
    const InterpolatorType::Pointer interpolator = InterpolatorType::New();
    const OptimizerType::Pointer optimizer = OptimizerType::New();
    const ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    const FilterType_Resample_3D::Pointer resampler = FilterType_Resample_3D::New();
    TransformType::Pointer initialTransform = TransformType::New();

    // Read images
    const ImageType3D::Pointer original = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename);
    const ImageType3D::Pointer original_scaled = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename);
    // MyITKImageHelper::showImage(original, "original");
    // MyITKImageHelper::showImage(original_scaled, "original_scaled");

    ImageType3D::DirectionType direction = original->GetDirection();
    // direction[0][0]=1;
    // direction[1][1]=1;
    // direction[2][2]=1;
    // direction[0][1]=0;
    // direction[0][2]=0;
    // direction[1][0]=0;
    // direction[1][2]=0;
    // direction[2][0]=0;
    // direction[2][1]=0;
    // original->SetDirection(direction);
    // original_scaled->SetDirection(direction);

    // Generate test case: Alter image for registration
    ImageType3D::SpacingType spacing = original_scaled->GetSpacing();
    spacing[0] *= scale;
    spacing[1] *= scale;
    original_scaled->SetSpacing(spacing);

    TransformType::FixedParametersType originalParameters = initialTransform->GetFixedParameters();
    TransformType::FixedParametersType originalParameters_extended = initialTransform->GetFixedParameters();
    const unsigned int N_originalParameters = originalParameters.GetSize();
    originalParameters_extended.SetSize(N_originalParameters+9);

    // Copy previous original parameters
    for (int i = 0; i < N_originalParameters; ++i)
    {
        originalParameters_extended[i] = originalParameters[i];
    }
    // Fill extended original parameters with direction information
    for (int i = 0; i < dimension; ++i)
    {
        for (int j = 0; j < dimension; ++j)
        {
            originalParameters_extended[N_originalParameters+dimension*i+j] = direction[i][j];
        }
    }
    initialTransform->SetFixedParameters(originalParameters_extended);

    // Initialize the registration_transform
    TransformInitializerType::Pointer initializer = TransformInitializerType::New();
    initializer->SetTransform(initialTransform);
    initializer->SetFixedImage( original );
    initializer->SetMovingImage( original_scaled );
    initializer->GeometryOn();
    // initializer->MomentsOn();
    initializer->InitializeTransform();
    // initialTransform->Print(std::cout);

    registration->SetInitialTransform( initialTransform );
    registration->SetFixedInitialTransform( EulerTransformType::New() ); // Otherwise segmentation fault
    registration->InPlaceOn();

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
    registration->SetFixedImage(original);
    registration->SetMovingImage(original_scaled);
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
    TransformType::ConstPointer registration_transform = registration->GetTransform();    
    // registration_transform->Print(std::cout);
    // std::cout << registration_transform->GetParameters() << std::endl;
    // MyITKImageHelper::printTransform(registration_transform);

    // Resample Image Filter
    resampler->SetOutputParametersFromImage(original);
    resampler->SetDefaultPixelValue( 0.0 );
    resampler->SetInterpolator( InterpolatorType::New() );
    resampler->SetInput(original_scaled);
    resampler->Update();
    const ImageType3D::Pointer original_scaled_resampled = resampler->GetOutput();
    original_scaled_resampled->DisconnectPipeline();

    resampler->SetTransform( registration_transform );
    resampler->Update();

    const ImageType3D::Pointer original_recovered = resampler->GetOutput();
    original_recovered->DisconnectPipeline();

    std::vector<ImageType3D::Pointer> image_vector;
    std::vector<std::string> title_vector;
    image_vector.push_back(original);
    image_vector.push_back(original_scaled_resampled);
    image_vector.push_back(original_recovered);
    title_vector.push_back("original");
    title_vector.push_back("original_scaled");
    title_vector.push_back("original_recovered");
    // std::string titles_array[7] = {"original", "original_scaled", "orignal_recovered"};
    // MyITKImageHelper::showImage(image_vector, titles_array);

    // Check accuracy
    const double scale_estimated = registration_transform->GetScale();
    const double abs_diff = std::abs(scale-scale_estimated);
    std::cout << "|scale - scale_est|  = " << abs_diff << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
    absoluteDifferenceImageFilter_3D->SetInput1(original_recovered);


    // Testing: InplaneSimilarity3DTransform
    TransformType::Pointer trafo = TransformType::New();
    // Always SetFixedParameters before SetParameters!
    TransformType::FixedParametersType originalParameterInplaneSim = registration_transform->GetFixedParameters();
    TransformType::ParametersType parameterInplaneSim = registration_transform->GetParameters();
    trafo->SetFixedParameters(originalParameterInplaneSim);
    trafo->SetParameters(parameterInplaneSim);
    resampler->SetTransform(trafo);
    resampler->Update();
    const ImageType3D::Pointer res_InplaneSim = resampler->GetOutput();
    res_InplaneSim->DisconnectPipeline();
    image_vector.push_back(res_InplaneSim);
    title_vector.push_back("original_InplaneSim");
    registration_transform->Print(std::cout);
    // trafo->Print(std::cout);
    absoluteDifferenceImageFilter_3D->SetInput2(res_InplaneSim);
    statisticsImageFilter_3D->SetInput(absoluteDifferenceImageFilter_3D->GetOutput());
    statisticsImageFilter_3D->Update();
    const double absDiff_InplaneSim = statisticsImageFilter_3D->GetSum();
    CHECK( absDiff_InplaneSim  == Approx(0).epsilon(tolerance));

    // Testing: AffineTransform
    AffineTransformType::Pointer affine = AffineTransformType::New();
    affine->SetTranslation(registration_transform->GetTranslation());
    affine->SetCenter(registration_transform->GetCenter());
    affine->SetMatrix(registration_transform->GetMatrix());
    resampler->SetTransform(affine);
    resampler->Update();
    const ImageType3D::Pointer res_affine = resampler->GetOutput();
    res_affine->DisconnectPipeline();
    image_vector.push_back(res_affine);
    title_vector.push_back("original_affine");
    absoluteDifferenceImageFilter_3D->SetInput2(res_affine);
    statisticsImageFilter_3D->SetInput(absoluteDifferenceImageFilter_3D->GetOutput());
    statisticsImageFilter_3D->Update();
    const double absDiff_affine = statisticsImageFilter_3D->GetSum();
    CHECK( absDiff_affine  == Approx(0).epsilon(tolerance));
    affine->Print(std::cout);

    // Testing: Euler3DTransform
    // EulerTransformType::Pointer euler = EulerTransformType::New();
    // EulerTransformType::ParametersType parameter = euler->GetParameters();
    // EulerTransformType::FixedParametersType center = euler->GetFixedParameters();
    // std::cout << euler->GetOffset() << std::endl;
    // // std::cout << parameterInplaneSim << std::endl;
    // // std::cout << originalParameterInplaneSim << std::endl;
    // for (int i = 0; i < 6; ++i)
    // {
    //     parameter[i] = parameterInplaneSim[i];
    // }
    // for (int i = 0; i < 3; ++i)
    // {
    //     center[i] = originalParameterInplaneSim[i];
    // }
    // euler->SetParameters(parameter);
    // euler->SetFixedParameters(center);
    // euler->Print(std::cout);

    // ImageType3D::SpacingType spacing_original = original->GetSpacing();
    // spacing_original[0] *= scale_estimated;    
    // spacing_original[1] *= scale_estimated;
    // original->SetSpacing(spacing_original);

    // resampler->SetOutputParametersFromImage(original);
    // resampler->SetTransform(euler);
    // resampler->UpdateLargestPossibleRegion();
    // resampler->Update();
    // const ImageType3D::Pointer out_2 = resampler->GetOutput();
    // out_2->DisconnectPipeline();
    // image_vector.push_back(out_2);
    // titles_array[4] = "array_original_scaledWarped_euler";
    MyITKImageHelper::showImage(image_vector, title_vector);
}
