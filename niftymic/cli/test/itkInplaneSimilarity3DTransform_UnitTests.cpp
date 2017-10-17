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
#include <itkImageDuplicator.h>

#include <itkSimilarity3DTransform.h>

#include <itkEuler3DTransform.h>
#include <itkVersorRigid3DTransform.h>
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

typedef itk::ImageDuplicator< ImageType3D > DuplicatorType_3D;

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
typedef itk::InplaneSimilarity3DTransform< PixelType > InplaneSimilarityTransformType;
typedef itk::VersorRigid3DTransform< PixelType > VersorRigid3DTransformType;

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
// typedef itk::RegistrationParameterScalesFromPhysicalShift< MetricType > ScalesEstimatorType;
typedef itk::RegistrationParameterScalesFromIndexShift< MetricType > ScalesEstimatorType;

// Unit tests
TEST_CASE( "itkInplaneSimilarity3DTransform: Brain", 
  "[itkInplaneSimilarity3DTransform: Brain]") {


    const double tolerance = 1e-3;
    const double tolerance_translation = 1e-1;
    const double scale = 0.9;
    TransformType::ParametersType parameters(6);
    parameters[0] = 0.01;
    parameters[1] = -0.05;
    parameters[2] = 0.02;
    parameters[3] = 2;
    parameters[4] = -5;
    parameters[5] = 10;

    const unsigned int dimension = 3;

    // Define input and output
    const std::string dir_input = "../test-data/";
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
    const ImageType3D::Pointer original_altered = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename);
    // MyITKImageHelper::showImage(original, "original");
    // MyITKImageHelper::showImage(original_altered, "original_altered");

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
    // original_altered->SetDirection(direction);

    /** Generate test case: Alter image for registration */
    ImageType3D::SpacingType spacing_new = original->GetSpacing();
    spacing_new[0] *= scale;
    spacing_new[1] *= scale;
    original_altered->SetSpacing(spacing_new);

    VersorRigid3DTransformType::Pointer motion = VersorRigid3DTransformType::New();
    motion->SetParameters(parameters);
    itk::Matrix<double,dimension,dimension> Lambda;
    Lambda.SetIdentity();
    Lambda(0,0) = scale;
    Lambda(1,1) = scale;
    original_altered->SetDirection(motion->GetMatrix() * direction);
    // VersorRigid3DTransformType::MatrixType matrix_inv = itk::Matrix<double,dimension,dimension>(motion->GetMatrix().GetInverse());
    original_altered->SetOrigin(motion->GetMatrix() * direction * Lambda * direction.GetInverse() * original->GetOrigin() + motion->GetTranslation());
    // motion->Print(std::cout);

    /** Recover transform */
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
    initializer->SetMovingImage( original_altered );
    // initializer->GeometryOn();
    initializer->MomentsOn();
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
    optimizer->SetGradientMagnitudeTolerance( 1e-5 ); //
    // optimizer->SetMinimumStepLength( 1e-6 );
    optimizer->SetScalesEstimator( scalesEstimator );
    // optimizer->SetDoEstimateLearningRateOnce( false );
    // optimizer->EstimateLearningRate();

    // Set registration
    registration->SetFixedImage(original);
    registration->SetMovingImage(original_altered);
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
    resampler->SetInput(original_altered);
    resampler->SetOutputParametersFromImage(original);
    resampler->SetDefaultPixelValue( 0.0 );
    resampler->SetInterpolator( InterpolatorType::New() );
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
    title_vector.push_back("original_altered");
    title_vector.push_back("original_recovered");
    // std::string titles_array[7] = {"original", "original_altered", "orignal_recovered"};
    // MyITKImageHelper::showImage(image_vector, titles_array);

    // Check accuracy
    const double scale_estimated = registration_transform->GetScale();
    const TransformType::ParametersType parameters_estimated = registration_transform->GetParameters();
    const TransformType::OffsetType offset_estimated = registration_transform->GetOffset();

    const double abs_diff_scale = std::abs(scale-scale_estimated);
    const double abs_diff_rotation_0 = std::abs(parameters[0]-parameters_estimated[0]);
    const double abs_diff_rotation_1 = std::abs(parameters[1]-parameters_estimated[1]);
    const double abs_diff_rotation_2 = std::abs(parameters[2]-parameters_estimated[2]);
    const double abs_diff_translation_0 = std::abs(parameters[3]-offset_estimated[0]);
    const double abs_diff_translation_1 = std::abs(parameters[4]-offset_estimated[1]);
    const double abs_diff_translation_2 = std::abs(parameters[5]-offset_estimated[2]);
    std::cout << "|scale - scale_estimated|  = " << abs_diff_scale << std::endl;
    std::cout << "|rotation - rotation_estimated| (versors) = (" << abs_diff_rotation_0 << ", " << abs_diff_rotation_0 << ", " << abs_diff_rotation_2 << ") " << std::endl;
    std::cout << "|translation - offset_estimated| = " << abs_diff_translation_0 << ", " << abs_diff_translation_1 << ", " << abs_diff_translation_2 << ") " << std::endl;
    
    CHECK( abs_diff_scale == Approx(0).epsilon(tolerance));
    CHECK( abs_diff_rotation_0 == Approx(0).epsilon(tolerance));
    CHECK( abs_diff_rotation_1 == Approx(0).epsilon(tolerance));
    CHECK( abs_diff_rotation_2 == Approx(0).epsilon(tolerance));
    CHECK( abs_diff_translation_0 == Approx(0).epsilon(tolerance_translation));
    CHECK( abs_diff_translation_1 == Approx(0).epsilon(tolerance_translation));
    CHECK( abs_diff_translation_2 == Approx(0).epsilon(tolerance_translation));

    /** Testing: InplaneSimilarity3DTransform */
    TransformType::Pointer inplaneSim = TransformType::New();
    // Always SetFixedParameters before SetParameters!
    inplaneSim->SetFixedParameters(registration_transform->GetFixedParameters());
    inplaneSim->SetParameters(registration_transform->GetParameters());
    // inplaneSim->Print(std::cout);

    resampler->SetTransform(inplaneSim);
    resampler->Update();
    const ImageType3D::Pointer res_InplaneSim = resampler->GetOutput();
    res_InplaneSim->DisconnectPipeline();
    image_vector.push_back(res_InplaneSim);
    title_vector.push_back("original_InplaneSim");
    registration_transform->Print(std::cout);
    absoluteDifferenceImageFilter_3D->SetInput1(original_recovered);
    absoluteDifferenceImageFilter_3D->SetInput2(res_InplaneSim);
    statisticsImageFilter_3D->SetInput(absoluteDifferenceImageFilter_3D->GetOutput());
    statisticsImageFilter_3D->Update();
    const double absDiff_InplaneSim = statisticsImageFilter_3D->GetSum();
    CHECK( absDiff_InplaneSim  == Approx(0).epsilon(tolerance));

    /** Testing: AffineTransform */
    AffineTransformType::Pointer affine = AffineTransformType::New();
    affine->SetTranslation(registration_transform->GetTranslation());
    affine->SetCenter(registration_transform->GetCenter());
    affine->SetMatrix(registration_transform->GetMatrix());
    // affine->Print(std::cout);
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

    /** Testing: VersorRigid3DTransform applied on scaled reference image */
    // Apply scaling directly to reference image 
    DuplicatorType_3D::Pointer imageDuplicator = DuplicatorType_3D::New();
    imageDuplicator->SetInputImage(original);
    imageDuplicator->Update();
    ImageType3D::Pointer original_scaled_estimate = imageDuplicator->GetOutput();
    original_scaled_estimate->DisconnectPipeline();

    ImageType3D::SpacingType spacing = original->GetSpacing();
    spacing[0] *= scale_estimated;
    spacing[1] *= scale_estimated;
    original_scaled_estimate->SetSpacing(spacing);

    // Create rigid transform to correct for the rigid motion estimated by InplaneSimilarity3DTransform
    // (Scaling leads to updated translation parameter (center=0); 
    //  alternatively, update center and translation part accordingly)
    VersorRigid3DTransformType::Pointer rigid = VersorRigid3DTransformType::New();
    VersorRigid3DTransformType::TranslationType translation = registration_transform->GetTranslation();
    VersorRigid3DTransformType::CenterType center = registration_transform->GetCenter();
    ImageType3D::PointType origin = original->GetOrigin();
    
    // Set Rotation
    rigid->SetParameters(registration_transform->GetParameters());
    
    // Set Translation
    VersorRigid3DTransformType::MatrixType R = rigid->GetMatrix();
    VersorRigid3DTransformType::MatrixType Lambda_estimated;

    Lambda_estimated.SetIdentity();
    Lambda_estimated(0,0) = scale_estimated;
    Lambda_estimated(1,1) = scale_estimated;
    translation += R*direction*Lambda_estimated*direction.GetInverse()*(origin-center);
    // Would like to add the following lines directly but I can't due to type
    // incompatibilities. Hence, this stupid workaround
    origin = R*origin;
    for (int i = 0; i < dimension; ++i)
    {
        translation[i] += center[i] - origin[i];
    }
    rigid->SetTranslation(translation);
    // rigid->Print(std::cout);

    // Resample 
    resampler->SetOutputParametersFromImage(original_scaled_estimate);
    // resampler->SetInput(original_scaled_estimate);
    resampler->SetTransform(rigid);
    resampler->Update();

    const ImageType3D::Pointer res_rigid = resampler->GetOutput();
    res_rigid->DisconnectPipeline();
    image_vector.push_back(res_rigid);
    title_vector.push_back("original_scaled_rigid");
    absoluteDifferenceImageFilter_3D->SetInput1(original_scaled_estimate);
    absoluteDifferenceImageFilter_3D->SetInput2(res_rigid);
    statisticsImageFilter_3D->SetInput(absoluteDifferenceImageFilter_3D->GetOutput());
    statisticsImageFilter_3D->Update();
    const double absDiff_rigid = statisticsImageFilter_3D->GetSum();

    absoluteDifferenceImageFilter_3D->SetInput1(original);
    absoluteDifferenceImageFilter_3D->SetInput2(original_recovered);
    statisticsImageFilter_3D->SetInput(absoluteDifferenceImageFilter_3D->GetOutput());
    statisticsImageFilter_3D->Update();
    const double absDiff_registration = statisticsImageFilter_3D->GetSum();

    CHECK( std::abs(absDiff_rigid - absDiff_registration) == Approx(0).epsilon(tolerance));

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
    // MyITKImageHelper::showImage(image_vector, title_vector);
}
