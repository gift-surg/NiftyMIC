/*! \brief Code to verify the implementation of itkAdjointOrientedGaussianInterpolateImageFilter.
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
#include <chrono>
// #include <iostream>
// #include <stdio.h>

#include <itkImage.h>
#include <itkResampleImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkAbsoluteValueDifferenceImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkEuler3DTransform.h>
#include <itkGradientImageFilter.h>
#include <itkGradientRecursiveGaussianImageFilter.h>
#include <itkDerivativeImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

// My includes
#include "MyITKImageHelper.h"
#include "itkAdjointOrientedGaussianInterpolateImageFilter.h"
#include "itkOrientedGaussianInterpolateImageFilter.h"
#include "itkOrientedGaussianInterpolateImageFunction.h"
#include "itkGradientEuler3DTransformImageFilter.h"
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

typedef itk::AbsoluteValueDifferenceImageFilter< ImageType2D, ImageType2D, ImageType2D> AbsoluteValueDifferenceImageFilter_2D;
typedef itk::AbsoluteValueDifferenceImageFilter< ImageType3D, ImageType3D, ImageType3D> AbsoluteValueDifferenceImageFilter_3D;

typedef itk::StatisticsImageFilter<ImageType2D> StatisticsImageFilterType_2D;
typedef itk::StatisticsImageFilter<ImageType3D> StatisticsImageFilterType_3D;

typedef itk::Euler3DTransform< PixelType > EulerTransformType;

typedef itk::GradientImageFilter<ImageType2D, PixelType, PixelType> FilterType_Gradient_2D;
typedef itk::GradientImageFilter<ImageType3D, PixelType, PixelType> FilterType_Gradient_3D;

typedef itk::DerivativeImageFilter<ImageType2D, ImageType2D> FilterType_Derivative_2D;
typedef itk::DerivativeImageFilter<ImageType3D, ImageType3D> FilterType_Derivative_3D;

typedef itk::GradientEuler3DTransformImageFilter<ImageType3D, PixelType, PixelType> FilterType_GradientEuler_3D;

// Unit tests

TEST_CASE( "Check 2D itkAdjointOrientedGaussianInterpolateImageFilter: Cross", 
  "[AdjointOrientedGaussian 2D: Cross]") {

    // Define input and output
    const std::string dir_input = "../test-data/";

    // const std::string filename_image_2D = "2D_SingleDot_50.nii.gz";
    const std::string filename_image_2D = "2D_Cross_50.nii.gz";
    // const std::string filename_image_2D = "2D_Text.nii.gz";
    // const std::string filename_image_2D = "2D_Lena_512.nii.gz";
    // const std::string filename_image_2D = "2D_SheppLoganPhantom_512.nii.gz";
    // const std::string filename_image_2D = "BrainWeb_2D.nii.gz";

    const double tolerance = 1e-6;

    // Set filter parameters
    const double alpha = 1;
    
    itk::Vector<double, 2> Sigma_2D;
    Sigma_2D[0] = 3;
    Sigma_2D[1] = 2;

    // Read images
    const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

    // Adjoint Oriented Gaussian Interpolate Image Filter
    const FilterType_AdjointOrientedGaussian_2D::Pointer filter_AdjointOrientedGaussian_2D = FilterType_AdjointOrientedGaussian_2D::New();

    filter_AdjointOrientedGaussian_2D->SetInput(image_2D);
    filter_AdjointOrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_AdjointOrientedGaussian_2D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_2D->SetSigma(Sigma_2D);
    filter_AdjointOrientedGaussian_2D->Update();

    // Resample Image Filter
    const InterpolatorType_2D::Pointer interpolator_2D = InterpolatorType_2D::New();
    interpolator_2D->SetSigma(Sigma_2D);
    interpolator_2D->SetAlpha(alpha);

    const  FilterType_Resample_2D::Pointer filter_Resample_2D = FilterType_Resample_2D::New();
    filter_Resample_2D->SetInput(image_2D);
    filter_Resample_2D->SetOutputParametersFromImage(image_2D);
    filter_Resample_2D->SetInterpolator(interpolator_2D);
    filter_Resample_2D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_2D->Update();  

    // Filters to evaluate absolute difference
    const MultiplyImageFilter_2D::Pointer multiplyFilter_2D = MultiplyImageFilter_2D::New();
    const AbsoluteValueDifferenceImageFilter_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilter_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilter_2D = StatisticsImageFilterType_2D::New();

    // Compute LHS (Ax,y), x=y=image_2D
    multiplyFilter_2D->SetInput1( filter_Resample_2D->GetOutput() );
    multiplyFilter_2D->SetInput2( image_2D );
    const ImageType2D::Pointer LHS = multiplyFilter_2D->GetOutput();

    statisticsImageFilter_2D->SetInput( LHS );
    statisticsImageFilter_2D->Update();
    const double sum_LHS = statisticsImageFilter_2D->GetSum();
    LHS->DisconnectPipeline(); // "I don't listen to what happens up my stream!"

    // Compute RHS (x,A'y), x=y=image_2D
    multiplyFilter_2D->SetInput1( image_2D );
    multiplyFilter_2D->SetInput2( filter_AdjointOrientedGaussian_2D->GetOutput() );
    const ImageType2D::Pointer RHS = multiplyFilter_2D->GetOutput();
    
    statisticsImageFilter_2D->SetInput( RHS );
    statisticsImageFilter_2D->Update();
    const double sum_RHS = statisticsImageFilter_2D->GetSum();
    RHS->DisconnectPipeline();

    // compute |(Ax,y) - (x,A'y)|
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // std::cout << "Check 2D itkAdjointOrientedGaussianInterpolateImageFilter: Cross" << std::endl;
    // std::cout << "\t|(Ax,y) - (x,A'y)| = " << abs_diff << std::endl;
    // std::cout << "\t|(Ax,y) - (x,A'y)|/(Ax,y) = " << std::abs(sum_LHS-sum_RHS)/sum_LHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

TEST_CASE( "Check 2D itkAdjointOrientedGaussianInterpolateImageFilter: Text", 
  "[AdjointOrientedGaussian 2D: Text]") {

    // Define input and output
    const std::string dir_input = "../test-data/";

    const std::string filename_image_2D = "2D_Text.nii.gz";

    const double tolerance = 1e-6;

    // Set filter parameters
    const double alpha = 1;
    
    itk::Vector<double, 2> Sigma_2D;
    Sigma_2D[0] = 3;
    Sigma_2D[1] = 2;

    // Read images
    const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

    // Adjoint Oriented Gaussian Interpolate Image Filter
    const FilterType_AdjointOrientedGaussian_2D::Pointer filter_AdjointOrientedGaussian_2D = FilterType_AdjointOrientedGaussian_2D::New();

    filter_AdjointOrientedGaussian_2D->SetInput(image_2D);
    filter_AdjointOrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_AdjointOrientedGaussian_2D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_2D->SetSigma(Sigma_2D);
    filter_AdjointOrientedGaussian_2D->Update();

    // Resample Image Filter
    const InterpolatorType_2D::Pointer interpolator_2D = InterpolatorType_2D::New();
    interpolator_2D->SetSigma(Sigma_2D);
    interpolator_2D->SetAlpha(alpha);

    const  FilterType_Resample_2D::Pointer filter_Resample_2D = FilterType_Resample_2D::New();
    filter_Resample_2D->SetInput(image_2D);
    filter_Resample_2D->SetOutputParametersFromImage(image_2D);
    filter_Resample_2D->SetInterpolator(interpolator_2D);
    filter_Resample_2D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_2D->Update();  

    // Filters to evaluate absolute difference
    const MultiplyImageFilter_2D::Pointer multiplyFilter_2D = MultiplyImageFilter_2D::New();
    const AbsoluteValueDifferenceImageFilter_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilter_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilter_2D = StatisticsImageFilterType_2D::New();

    // Compute LHS (Ax,y), x=y=image_2D
    multiplyFilter_2D->SetInput1( filter_Resample_2D->GetOutput() );
    multiplyFilter_2D->SetInput2( image_2D );
    const ImageType2D::Pointer LHS = multiplyFilter_2D->GetOutput();

    statisticsImageFilter_2D->SetInput( LHS );
    statisticsImageFilter_2D->Update();
    const double sum_LHS = statisticsImageFilter_2D->GetSum();
    LHS->DisconnectPipeline(); // "I don't listen to what happens up my stream!"

    // Compute RHS (x,A'y), x=y=image_2D
    multiplyFilter_2D->SetInput1( image_2D );
    multiplyFilter_2D->SetInput2( filter_AdjointOrientedGaussian_2D->GetOutput() );
    const ImageType2D::Pointer RHS = multiplyFilter_2D->GetOutput();
    
    statisticsImageFilter_2D->SetInput( RHS );
    statisticsImageFilter_2D->Update();
    const double sum_RHS = statisticsImageFilter_2D->GetSum();
    RHS->DisconnectPipeline();

    // compute |(Ax,y) - (x,A'y)|
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // std::cout << "Filter: |(Ax,y) - (x,A'y)| = " << abs_diff << std::endl;
    // std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    // std::cout << "        (x,A'y) = " << sum_RHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

TEST_CASE( "Check 2D itkAdjointOrientedGaussianInterpolateImageFilter: Shepp-Logan", 
  "[AdjointOrientedGaussian 2D: Shepp-Logan]") {

    // Define input and output
    const std::string dir_input = "../test-data/";

    const std::string filename_image_2D = "2D_SheppLoganPhantom_512.nii.gz";

    const double tolerance = 1e-6;

    // Set filter parameters
    const double alpha = 1;
    
    itk::Vector<double, 2> Sigma_2D;
    Sigma_2D[0] = 3;
    Sigma_2D[1] = 2;

    // Read images
    const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

    // Adjoint Oriented Gaussian Interpolate Image Filter
    const FilterType_AdjointOrientedGaussian_2D::Pointer filter_AdjointOrientedGaussian_2D = FilterType_AdjointOrientedGaussian_2D::New();

    filter_AdjointOrientedGaussian_2D->SetInput(image_2D);
    filter_AdjointOrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_AdjointOrientedGaussian_2D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_2D->SetSigma(Sigma_2D);
    filter_AdjointOrientedGaussian_2D->Update();

    // Resample Image Filter
    const InterpolatorType_2D::Pointer interpolator_2D = InterpolatorType_2D::New();
    interpolator_2D->SetSigma(Sigma_2D);
    interpolator_2D->SetAlpha(alpha);

    const  FilterType_Resample_2D::Pointer filter_Resample_2D = FilterType_Resample_2D::New();
    filter_Resample_2D->SetInput(image_2D);
    filter_Resample_2D->SetOutputParametersFromImage(image_2D);
    filter_Resample_2D->SetInterpolator(interpolator_2D);
    filter_Resample_2D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_2D->Update();  

    // Filters to evaluate absolute difference
    const MultiplyImageFilter_2D::Pointer multiplyFilter_2D = MultiplyImageFilter_2D::New();
    const AbsoluteValueDifferenceImageFilter_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilter_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilter_2D = StatisticsImageFilterType_2D::New();

    // Compute LHS (Ax,y), x=y=image_2D
    multiplyFilter_2D->SetInput1( filter_Resample_2D->GetOutput() );
    multiplyFilter_2D->SetInput2( image_2D );
    const ImageType2D::Pointer LHS = multiplyFilter_2D->GetOutput();

    statisticsImageFilter_2D->SetInput( LHS );
    statisticsImageFilter_2D->Update();
    const double sum_LHS = statisticsImageFilter_2D->GetSum();
    LHS->DisconnectPipeline(); // "I don't listen to what happens up my stream!"

    // Compute RHS (x,A'y), x=y=image_2D
    multiplyFilter_2D->SetInput1( image_2D );
    multiplyFilter_2D->SetInput2( filter_AdjointOrientedGaussian_2D->GetOutput() );
    const ImageType2D::Pointer RHS = multiplyFilter_2D->GetOutput();
    
    statisticsImageFilter_2D->SetInput( RHS );
    statisticsImageFilter_2D->Update();
    const double sum_RHS = statisticsImageFilter_2D->GetSum();
    RHS->DisconnectPipeline();

    // compute |(Ax,y) - (x,A'y)|
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // std::cout << "Filter: |(Ax,y) - (x,A'y)| = " << abs_diff << std::endl;
    // std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    // std::cout << "        (x,A'y) = " << sum_RHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

TEST_CASE( "Check 2D itkAdjointOrientedGaussianInterpolateImageFilter: BrainWeb", 
  "[AdjointOrientedGaussian 2D: BrainWeb]") {

    // Define input and output
    const std::string dir_input = "../test-data/";

    const std::string filename_image_2D = "2D_BrainWeb.nii.gz";

    const double tolerance = 1e-6;

    // Set filter parameters
    const double alpha = 1;
    
    itk::Vector<double, 2> Sigma_2D;
    Sigma_2D[0] = 3;
    Sigma_2D[1] = 2;

    // Read images
    const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

    // Adjoint Oriented Gaussian Interpolate Image Filter
    const FilterType_AdjointOrientedGaussian_2D::Pointer filter_AdjointOrientedGaussian_2D = FilterType_AdjointOrientedGaussian_2D::New();

    filter_AdjointOrientedGaussian_2D->SetInput(image_2D);
    filter_AdjointOrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_AdjointOrientedGaussian_2D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_2D->SetSigma(Sigma_2D);
    filter_AdjointOrientedGaussian_2D->Update();

    // Resample Image Filter
    const InterpolatorType_2D::Pointer interpolator_2D = InterpolatorType_2D::New();
    interpolator_2D->SetSigma(Sigma_2D);
    interpolator_2D->SetAlpha(alpha);

    const  FilterType_Resample_2D::Pointer filter_Resample_2D = FilterType_Resample_2D::New();
    filter_Resample_2D->SetInput(image_2D);
    filter_Resample_2D->SetOutputParametersFromImage(image_2D);
    filter_Resample_2D->SetInterpolator(interpolator_2D);
    filter_Resample_2D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_2D->Update();  

    // Filters to evaluate absolute difference
    const MultiplyImageFilter_2D::Pointer multiplyFilter_2D = MultiplyImageFilter_2D::New();
    const AbsoluteValueDifferenceImageFilter_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilter_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilter_2D = StatisticsImageFilterType_2D::New();

    // Compute LHS (Ax,y), x=y=image_2D
    multiplyFilter_2D->SetInput1( filter_Resample_2D->GetOutput() );
    multiplyFilter_2D->SetInput2( image_2D );
    const ImageType2D::Pointer LHS = multiplyFilter_2D->GetOutput();
    
    statisticsImageFilter_2D->SetInput( LHS );
    statisticsImageFilter_2D->Update();
    const double sum_LHS = statisticsImageFilter_2D->GetSum();
    LHS->DisconnectPipeline(); // "I don't listen to what happens up my stream!"

    // Compute RHS (x,A'y), x=y=image_2D
    multiplyFilter_2D->SetInput1( image_2D );
    multiplyFilter_2D->SetInput2( filter_AdjointOrientedGaussian_2D->GetOutput() );
    const ImageType2D::Pointer RHS = multiplyFilter_2D->GetOutput();
    
    statisticsImageFilter_2D->SetInput( RHS );
    statisticsImageFilter_2D->Update();
    const double sum_RHS = statisticsImageFilter_2D->GetSum();
    RHS->DisconnectPipeline();

    // compute |(Ax,y) - (x,A'y)|
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput());

    // std::cout << "Filter: |(Ax,y) - (x,A'y)| = " << abs_diff << std::endl;
    // std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    // std::cout << "        (x,A'y) = " << sum_RHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}


TEST_CASE( "Check 3D itkAdjointOrientedGaussianInterpolateImageFilter: Single Dot", 
  "[AdjointOrientedGaussian 3D: Single Dot]") {

    // Define input and output
    const std::string dir_input = "../test-data/";

    const std::string filename_HR_volume = "3D_SingleDot_50.nii.gz";
    const std::string filename_slice = "3D_SingleDot_50.nii.gz";

    const double tolerance = 1e-6;

    // Set filter parameters
    const double alpha = 2;
    
    itk::Vector<double, 9> Cov_3D;
    Cov_3D.Fill(0);
    Cov_3D[0] = 9;
    Cov_3D[4] = 4;
    Cov_3D[8] = 1;

    // Read images
    const ImageType3D::Pointer HR_volume = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_HR_volume);
    const ImageType3D::Pointer slice = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_slice);

    // Adjoint Oriented Gaussian Interpolate Image Filter
    const FilterType_AdjointOrientedGaussian_3D::Pointer filter_AdjointOrientedGaussian_3D = FilterType_AdjointOrientedGaussian_3D::New();

    filter_AdjointOrientedGaussian_3D->SetInput(slice);
    filter_AdjointOrientedGaussian_3D->SetOutputParametersFromImage(HR_volume);
    filter_AdjointOrientedGaussian_3D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_3D->SetCovariance(Cov_3D);

    filter_AdjointOrientedGaussian_3D->Update();

    // Resample Image Filter
    const InterpolatorType_3D::Pointer interpolator_3D = InterpolatorType_3D::New();
    interpolator_3D->SetCovariance(Cov_3D);
    interpolator_3D->SetAlpha(alpha);

    const  FilterType_Resample_3D::Pointer filter_Resample_3D = FilterType_Resample_3D::New();
    filter_Resample_3D->SetInput(HR_volume);
    filter_Resample_3D->SetOutputParametersFromImage(slice);
    filter_Resample_3D->SetInterpolator(interpolator_3D);
    filter_Resample_3D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_3D->Update();  

    // Filters to evaluate absolute difference
    const MultiplyImageFilter_3D::Pointer multiplyFilter_3D = MultiplyImageFilter_3D::New();
    const AbsoluteValueDifferenceImageFilter_3D::Pointer absDiffFilter_3D = AbsoluteValueDifferenceImageFilter_3D::New();
    const StatisticsImageFilterType_3D::Pointer statisticsImageFilter_3D = StatisticsImageFilterType_3D::New();

    // Compute LHS (Ax,y)
    multiplyFilter_3D->SetInput1( filter_Resample_3D->GetOutput() );
    multiplyFilter_3D->SetInput2( slice );
    const ImageType3D::Pointer LHS = multiplyFilter_3D->GetOutput();

    statisticsImageFilter_3D->SetInput( LHS );
    statisticsImageFilter_3D->Update();
    const double sum_LHS = statisticsImageFilter_3D->GetSum();
    LHS->DisconnectPipeline(); // "I don't listen to what happens up my stream!"

    // Compute RHS (x,A'y), x=y=image_3D
    multiplyFilter_3D->SetInput1( HR_volume );
    multiplyFilter_3D->SetInput2( filter_AdjointOrientedGaussian_3D->GetOutput() );
    const ImageType3D::Pointer RHS = multiplyFilter_3D->GetOutput();
    
    statisticsImageFilter_3D->SetInput( RHS );
    statisticsImageFilter_3D->Update();
    const double sum_RHS = statisticsImageFilter_3D->GetSum();
    RHS->DisconnectPipeline();

    // compute |(Ax,y) - (x,A'y)|
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput());

    // std::cout << "Filter: |(Ax,y) - (x,A'y)| = " << abs_diff << std::endl;
    // std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    // std::cout << "        (x,A'y) = " << sum_RHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

TEST_CASE( "Check 3D itkAdjointOrientedGaussianInterpolateImageFilter: Cross", 
  "[AdjointOrientedGaussian 3D: Cross]") {

    // Define input and output
    const std::string dir_input = "../test-data/";

    const std::string filename_HR_volume = "3D_Cross_50.nii.gz";
    const std::string filename_slice = "3D_Cross_50.nii.gz";

    const double tolerance = 1e-6;

    // Set filter parameters
    const double alpha = 2;
    
    itk::Vector<double, 9> Cov_3D;
    Cov_3D.Fill(0);
    Cov_3D[0] = 9;
    Cov_3D[4] = 4;
    Cov_3D[8] = 1;

    // Read images
    const ImageType3D::Pointer HR_volume = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_HR_volume);
    const ImageType3D::Pointer slice = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_slice);

    // Adjoint Oriented Gaussian Interpolate Image Filter
    // Measure time: Start
    auto start = std::chrono::system_clock::now();

    const FilterType_AdjointOrientedGaussian_3D::Pointer filter_AdjointOrientedGaussian_3D = FilterType_AdjointOrientedGaussian_3D::New();

    filter_AdjointOrientedGaussian_3D->SetInput(slice);
    filter_AdjointOrientedGaussian_3D->SetOutputParametersFromImage(HR_volume);
    filter_AdjointOrientedGaussian_3D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_3D->SetCovariance(Cov_3D);
    filter_AdjointOrientedGaussian_3D->Update();

    // Measure time: Stop
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    // std::cout << "3D Cross" << std::endl;
    // std::cout << "\tElapsed time (Adjoint Operator): " << diff.count() << " s" << std::endl;

    // Resample Image Filter
    // Measure time: Start
    start = std::chrono::system_clock::now();

    const InterpolatorType_3D::Pointer interpolator_3D = InterpolatorType_3D::New();
    interpolator_3D->SetCovariance(Cov_3D);
    interpolator_3D->SetAlpha(alpha);

    const  FilterType_Resample_3D::Pointer filter_Resample_3D = FilterType_Resample_3D::New();
    filter_Resample_3D->SetInput(HR_volume);
    filter_Resample_3D->SetOutputParametersFromImage(slice);
    filter_Resample_3D->SetInterpolator(interpolator_3D);
    filter_Resample_3D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_3D->Update();  

    // Measure time: Stop
    end = std::chrono::system_clock::now();
    diff = end-start;
    // std::cout << "\tElapsed time (Forward Operator): " << diff.count() << " s" << std::endl;

    // Filters to evaluate absolute difference
    const MultiplyImageFilter_3D::Pointer multiplyFilter_3D = MultiplyImageFilter_3D::New();
    const AbsoluteValueDifferenceImageFilter_3D::Pointer absDiffFilter_3D = AbsoluteValueDifferenceImageFilter_3D::New();
    const StatisticsImageFilterType_3D::Pointer statisticsImageFilter_3D = StatisticsImageFilterType_3D::New();

    // Compute LHS (Ax,y)
    multiplyFilter_3D->SetInput1( filter_Resample_3D->GetOutput() );
    multiplyFilter_3D->SetInput2( slice );
    const ImageType3D::Pointer LHS = multiplyFilter_3D->GetOutput();

    statisticsImageFilter_3D->SetInput( LHS );
    statisticsImageFilter_3D->Update();
    const double sum_LHS = statisticsImageFilter_3D->GetSum();
    LHS->DisconnectPipeline(); // "I don't listen to what happens up my stream!"

    // Compute RHS (x,A'y), x=y=image_3D
    multiplyFilter_3D->SetInput1( HR_volume );
    multiplyFilter_3D->SetInput2( filter_AdjointOrientedGaussian_3D->GetOutput() );
    const ImageType3D::Pointer RHS = multiplyFilter_3D->GetOutput();
    
    statisticsImageFilter_3D->SetInput( RHS );
    statisticsImageFilter_3D->Update();
    const double sum_RHS = statisticsImageFilter_3D->GetSum();
    RHS->DisconnectPipeline();

    // compute |(Ax,y) - (x,A'y)|
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput());

    // std::cout << "3D-Cross: |(Ax,y) - (x,A'y)| = " << abs_diff << std::endl;
    // std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    // std::cout << "        (x,A'y) = " << sum_RHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}


TEST_CASE( "Check 3D itkAdjointOrientedGaussianInterpolateImageFilter: Real scenario but higher covariance", 
  "[AdjointOrientedGaussian 3D: Real scenario but higher covariance]") {

    // Define input and output
    const std::string dir_input = "../test-data/";

    const std::string filename_HR_volume = "FetalBrain_reconstruction_4stacks.nii.gz";
    const std::string filename_slice = "FetalBrain_stack2_registered_midslice.nii.gz";

    const double tolerance = 1e-6;

    // Set filter parameters
    const double alpha = 2;
    
    itk::Vector<double, 3> Sigma_3D;
    Sigma_3D[0] = 2;
    Sigma_3D[1] = 2;
    Sigma_3D[2] = 3;

    // Read images
    const ImageType3D::Pointer HR_volume = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_HR_volume);
    const ImageType3D::Pointer slice = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_slice);


    // Adjoint Oriented Gaussian Interpolate Image Filter
    // Measure time: Start
    auto start = std::chrono::system_clock::now();

    const FilterType_AdjointOrientedGaussian_3D::Pointer filter_AdjointOrientedGaussian_3D = FilterType_AdjointOrientedGaussian_3D::New();

    filter_AdjointOrientedGaussian_3D->SetInput(slice);
    filter_AdjointOrientedGaussian_3D->SetOutputParametersFromImage(HR_volume);
    filter_AdjointOrientedGaussian_3D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_3D->SetSigma(Sigma_3D);
    filter_AdjointOrientedGaussian_3D->Update();

    // Measure time: Stop
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    const ImageType3D::RegionType::SizeType size = HR_volume->GetBufferedRegion().GetSize();
    // std::cout << "3D Realistic Images (But higher covariance)" << std::endl;
    // std::cout << "\tSize of HR volume: " << size << std::endl;
    // std::cout << "\tNumber of pixels HR volume: " << size[0]*size[1]*size[2] << std::endl;
    // std::cout << "\tElapsed time (Adjoint Operator): " << diff.count() << " s" << std::endl;

    // Resample Image Filter
    // Measure time: Start
    start = std::chrono::system_clock::now();

    const InterpolatorType_3D::Pointer interpolator_3D = InterpolatorType_3D::New();
    interpolator_3D->SetSigma(Sigma_3D);
    interpolator_3D->SetAlpha(alpha);

    const  FilterType_Resample_3D::Pointer filter_Resample_3D = FilterType_Resample_3D::New();
    filter_Resample_3D->SetInput(HR_volume);
    filter_Resample_3D->SetOutputParametersFromImage(slice);
    filter_Resample_3D->SetInterpolator(interpolator_3D);
    filter_Resample_3D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_3D->Update();  

    // Measure time: Stop
    end = std::chrono::system_clock::now();
    diff = end-start;
    // std::cout << "\tElapsed time (Forward Operator): " << diff.count() << " s" << std::endl;

    // Filters to evaluate absolute difference
    const MultiplyImageFilter_3D::Pointer multiplyFilter_3D = MultiplyImageFilter_3D::New();
    const AbsoluteValueDifferenceImageFilter_3D::Pointer absDiffFilter_3D = AbsoluteValueDifferenceImageFilter_3D::New();
    const StatisticsImageFilterType_3D::Pointer statisticsImageFilter_3D = StatisticsImageFilterType_3D::New();

    // Compute LHS (Ax,y)
    multiplyFilter_3D->SetInput1( filter_Resample_3D->GetOutput() );
    multiplyFilter_3D->SetInput2( slice );
    const ImageType3D::Pointer LHS = multiplyFilter_3D->GetOutput();

    statisticsImageFilter_3D->SetInput( LHS );
    statisticsImageFilter_3D->Update();
    const double sum_LHS = statisticsImageFilter_3D->GetSum();
    LHS->DisconnectPipeline(); // "I don't listen to what happens up my stream!"

    // Compute RHS (x,A'y), x=y=image_3D
    multiplyFilter_3D->SetInput1( HR_volume );
    multiplyFilter_3D->SetInput2( filter_AdjointOrientedGaussian_3D->GetOutput() );
    const ImageType3D::Pointer RHS = multiplyFilter_3D->GetOutput();
    
    statisticsImageFilter_3D->SetInput( RHS );
    statisticsImageFilter_3D->Update();
    const double sum_RHS = statisticsImageFilter_3D->GetSum();
    RHS->DisconnectPipeline();

    // compute |(Ax,y) - (x,A'y)|
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput());

    // std::cout << "\t|(Ax,y) - (x,A'y)| = " << abs_diff << std::endl;
    // std::cout << "\t|(Ax,y) - (x,A'y)|/(Ax,y) = " << std::abs(sum_LHS-sum_RHS)/sum_LHS  << " (rel. error)" << std::endl;
    // std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    // std::cout << "        (x,A'y) = " << sum_RHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

TEST_CASE( "Check 3D itkAdjointOrientedGaussianInterpolateImageFilter: Real scenario", 
  "[AdjointOrientedGaussian 3D: Real scenario]") {

    // Define input and output
    const std::string dir_input = "../test-data/";

    const std::string filename_HR_volume = "FetalBrain_reconstruction_4stacks.nii.gz";
    const std::string filename_slice = "FetalBrain_stack2_registered_midslice.nii.gz";

    const double tolerance = 1e-6;

    // Set filter parameters
    const double alpha = 2;
    
    itk::Vector<double, 9> Cov_3D;
    Cov_3D.Fill(0);
    Cov_3D[0] = 0.26786367;
    Cov_3D[4] = 0.26786367;
    Cov_3D[8] = 2.67304559;

    // Read images
    const ImageType3D::Pointer HR_volume = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_HR_volume);
    const ImageType3D::Pointer slice = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_slice);

    // Adjoint Oriented Gaussian Interpolate Image Filter
    // Measure time: Start
    auto start = std::chrono::system_clock::now();

    const FilterType_AdjointOrientedGaussian_3D::Pointer filter_AdjointOrientedGaussian_3D = FilterType_AdjointOrientedGaussian_3D::New();

    filter_AdjointOrientedGaussian_3D->SetInput(slice);
    filter_AdjointOrientedGaussian_3D->SetOutputParametersFromImage(HR_volume);
    filter_AdjointOrientedGaussian_3D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_3D->SetCovariance(Cov_3D);
    filter_AdjointOrientedGaussian_3D->Update();

    // Measure time: Stop
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    const ImageType3D::RegionType::SizeType size = HR_volume->GetBufferedRegion().GetSize();
    std::cout << "3D Realistic Images" << std::endl;
    std::cout << "\tSize of HR volume: " << size << std::endl;
    std::cout << "\tNumber of pixels HR volume: " << size[0]*size[1]*size[2] << std::endl;
    std::cout << "\tElapsed time (Adjoint Operator): " << diff.count() << " s" << std::endl;

    // Resample Image Filter
    // Measure time: Start
    start = std::chrono::system_clock::now();

    const InterpolatorType_3D::Pointer interpolator_3D = InterpolatorType_3D::New();
    interpolator_3D->SetCovariance(Cov_3D);
    interpolator_3D->SetAlpha(alpha);

    const  FilterType_Resample_3D::Pointer filter_Resample_3D = FilterType_Resample_3D::New();
    filter_Resample_3D->SetInput(HR_volume);
    filter_Resample_3D->SetOutputParametersFromImage(slice);
    filter_Resample_3D->SetInterpolator(interpolator_3D);
    filter_Resample_3D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_3D->Update();  

    // Measure time: Stop
    end = std::chrono::system_clock::now();
    diff = end-start;
    std::cout << "\tElapsed time (Forward Operator): " << diff.count() << " s (Resample Filter + Gaussian Interpolator)" << std::endl;

    // Filters to evaluate absolute difference
    const MultiplyImageFilter_3D::Pointer multiplyFilter_3D = MultiplyImageFilter_3D::New();
    const AbsoluteValueDifferenceImageFilter_3D::Pointer absDiffFilter_3D = AbsoluteValueDifferenceImageFilter_3D::New();
    const StatisticsImageFilterType_3D::Pointer statisticsImageFilter_3D = StatisticsImageFilterType_3D::New();

    // Compute LHS (Ax,y)
    multiplyFilter_3D->SetInput1( filter_Resample_3D->GetOutput() );
    multiplyFilter_3D->SetInput2( slice );
    const ImageType3D::Pointer LHS = multiplyFilter_3D->GetOutput();

    statisticsImageFilter_3D->SetInput( LHS );
    statisticsImageFilter_3D->Update();
    const double sum_LHS = statisticsImageFilter_3D->GetSum();
    LHS->DisconnectPipeline(); // "I don't listen to what happens up my stream!"

    // Compute RHS (x,A'y), x=y=image_3D
    multiplyFilter_3D->SetInput1( HR_volume );
    multiplyFilter_3D->SetInput2( filter_AdjointOrientedGaussian_3D->GetOutput() );
    const ImageType3D::Pointer RHS = multiplyFilter_3D->GetOutput();
    
    statisticsImageFilter_3D->SetInput( RHS );
    statisticsImageFilter_3D->Update();
    const double sum_RHS = statisticsImageFilter_3D->GetSum();
    RHS->DisconnectPipeline();

    // compute |(Ax,y) - (x,A'y)|
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput());

    // std::cout << "\t|(Ax,y) - (x,A'y)| = " << abs_diff << std::endl;
    // std::cout << "\t|(Ax,y) - (x,A'y)|/(Ax,y) = " << std::abs(sum_LHS-sum_RHS)/sum_LHS  << " (rel. error)"  << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

