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
/*
TEST_CASE( "Check 2D itkAdjointOrientedGaussianInterpolateImageFilter: Cross", 
  "[AdjointOrientedGaussian 2D: Cross]") {

    // Define input and output
    const std::string dir_input = "../exampleData/";

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
    const std::string dir_input = "../exampleData/";

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
    const std::string dir_input = "../exampleData/";

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
    const std::string dir_input = "../exampleData/";

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
    const std::string dir_input = "../exampleData/";

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
    const std::string dir_input = "../exampleData/";

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
    const std::string dir_input = "../exampleData/";

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
    const std::string dir_input = "../exampleData/";

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

// Check weather itkOrientedGaussianInterpolateImageFilter yields the same 
// result as the one obtained via itkOrientedGaussianInterpolateImageFilter.
TEST_CASE( "Check 2D itkOrientedGaussianInterpolateImageFilter: BrainWeb", 
  "[OrientedGaussian 2D: BrainWeb]") {

    // Define input and output
    const std::string dir_input = "../exampleData/";

    const std::string filename_image_2D = "2D_BrainWeb.nii.gz";

    const double tolerance = 1e-6;

    // Set filter parameters
    const double alpha = 1;
    
    itk::Vector<double, 2> Sigma_2D;
    Sigma_2D[0] = 3;
    Sigma_2D[1] = 2;

    // Read images
    const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

    // Oriented Gaussian Interpolate Image Filter
    // Measure time: Start
    auto start = std::chrono::system_clock::now();

    const FilterType_OrientedGaussian_2D::Pointer filter_OrientedGaussian_2D = FilterType_OrientedGaussian_2D::New();

    filter_OrientedGaussian_2D->SetInput(image_2D);
    filter_OrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_OrientedGaussian_2D->SetAlpha(alpha);
    filter_OrientedGaussian_2D->SetSigma(Sigma_2D);
    filter_OrientedGaussian_2D->SetDefaultPixelValue( 0.0 );  
    filter_OrientedGaussian_2D->Update();

    // Measure time: Stop
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    const ImageType2D::RegionType::SizeType size = image_2D->GetBufferedRegion().GetSize();
    // std::cout << "Oriented Gaussian Filter test: 2D" << std::endl;
    // std::cout << "\tSize of HR volume: " << size << std::endl;
    // std::cout << "\tNumber of pixels HR volume: " << size[0]*size[1]*size[2] << std::endl;
    // std::cout << "\tElapsed time (Gaussian Filter): " << diff.count() << " s" << std::endl;

    // Resample Image Filter with Oriented Gaussian Interpolator
    // Measure time: Start
    start = std::chrono::system_clock::now();

    const InterpolatorType_2D::Pointer interpolator_2D = InterpolatorType_2D::New();
    interpolator_2D->SetSigma(Sigma_2D);
    interpolator_2D->SetAlpha(alpha);

    const  FilterType_Resample_2D::Pointer filter_Resample_2D = FilterType_Resample_2D::New();
    filter_Resample_2D->SetInput(image_2D);
    filter_Resample_2D->SetOutputParametersFromImage(image_2D);
    filter_Resample_2D->SetInterpolator(interpolator_2D);
    filter_Resample_2D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_2D->Update();  

    // Measure time: Stop
    end = std::chrono::system_clock::now();
    diff = end-start;
    // std::cout << "\tElapsed time (Resample Filter + Gaussian Interpolator): " << diff.count() << " s" << std::endl;

    // Filters to evaluate absolute difference
    const AbsoluteValueDifferenceImageFilter_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilter_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilter_2D = StatisticsImageFilterType_2D::New();

    // Compute LHS (Ax,y)
    absDiffFilter_2D->SetInput1( filter_OrientedGaussian_2D->GetOutput() );
    absDiffFilter_2D->SetInput2( filter_Resample_2D->GetOutput() );
    absDiffFilter_2D->Update();

    statisticsImageFilter_2D->SetInput( absDiffFilter_2D->GetOutput() );
    statisticsImageFilter_2D->Update();
    const double abs_diff = statisticsImageFilter_2D->GetSum();

    // MyITKImageHelper::showImage(filter_OrientedGaussian_2D->GetOutput(), filter_Resample_2D->GetOutput(), "GaussianFilter_InterpolateFilter");

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

TEST_CASE( "Check 3D itkOrientedGaussianInterpolateImageFilter: Real scenario", 
  "[OrientedGaussian 3D: Real scenario]") {

    // Define input and output
    const std::string dir_input = "../exampleData/";

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

    // Oriented Gaussian Interpolate Image Filter
    // Measure time: Start
    auto start = std::chrono::system_clock::now();

    const FilterType_OrientedGaussian_3D::Pointer filter_OrientedGaussian_3D = FilterType_OrientedGaussian_3D::New();

    filter_OrientedGaussian_3D->SetInput(HR_volume);
    filter_OrientedGaussian_3D->SetOutputParametersFromImage(slice);
    filter_OrientedGaussian_3D->SetAlpha(alpha);
    filter_OrientedGaussian_3D->SetCovariance(Cov_3D);
    filter_OrientedGaussian_3D->SetDefaultPixelValue( 0.0 );  
    filter_OrientedGaussian_3D->Update();

    // Measure time: Stop
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    const ImageType3D::RegionType::SizeType size = HR_volume->GetBufferedRegion().GetSize();
    std::cout << "Oriented Gaussian Filter test: 3D Realistic Images" << std::endl;
    std::cout << "\tSize of HR volume: " << size << std::endl;
    std::cout << "\tNumber of pixels HR volume: " << size[0]*size[1]*size[2] << std::endl;
    std::cout << "\tElapsed time (Gaussian Filter): " << diff.count() << " s" << std::endl;

    // Resample Image Filter with Oriented Gaussian Interpolator
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
    std::cout << "\tElapsed time (Resample Filter + Gaussian Interpolator): " << diff.count() << " s" << std::endl;

    // Filters to evaluate absolute difference
    const AbsoluteValueDifferenceImageFilter_3D::Pointer absDiffFilter_3D = AbsoluteValueDifferenceImageFilter_3D::New();
    const StatisticsImageFilterType_3D::Pointer statisticsImageFilter_3D = StatisticsImageFilterType_3D::New();

    // Compute LHS (Ax,y)
    absDiffFilter_3D->SetInput1( filter_OrientedGaussian_3D->GetOutput() );
    absDiffFilter_3D->SetInput2( filter_Resample_3D->GetOutput() );
    absDiffFilter_3D->Update();

    statisticsImageFilter_3D->SetInput( absDiffFilter_3D->GetOutput() );
    statisticsImageFilter_3D->Update();
    const double abs_diff = statisticsImageFilter_3D->GetSum();

    // MyITKImageHelper::showImage(filter_OrientedGaussian_3D->GetOutput(), filter_Resample_3D->GetOutput(), "GaussianFilter_InterpolateFilter");

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

TEST_CASE( "Check 3D itkOrientedGaussianInterpolateImageFilter: Real scenario but higher covariance", 
  "[OrientedGaussian 3D: Real scenario but higher covariance]") {

    // Define input and output
    const std::string dir_input = "../exampleData/";

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

    // Oriented Gaussian Interpolate Image Filter
    // Measure time: Start
    auto start = std::chrono::system_clock::now();

    const FilterType_OrientedGaussian_3D::Pointer filter_OrientedGaussian_3D = FilterType_OrientedGaussian_3D::New();

    filter_OrientedGaussian_3D->SetInput(HR_volume);
    filter_OrientedGaussian_3D->SetOutputParametersFromImage(slice);
    filter_OrientedGaussian_3D->SetAlpha(alpha);
    filter_OrientedGaussian_3D->SetSigma(Sigma_3D);
    filter_OrientedGaussian_3D->SetDefaultPixelValue( 0.0 );  
    filter_OrientedGaussian_3D->Update();

    // Measure time: Stop
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    const ImageType3D::RegionType::SizeType size = HR_volume->GetBufferedRegion().GetSize();
    // std::cout << "Oriented Gaussian Filter test: 3D Realistic Images" << std::endl;
    // std::cout << "\tSize of HR volume: " << size << std::endl;
    // std::cout << "\tNumber of pixels HR volume: " << size[0]*size[1]*size[2] << std::endl;
    // std::cout << "\tElapsed time (Gaussian Filter): " << diff.count() << " s" << std::endl;

    // Resample Image Filter with Oriented Gaussian Interpolator
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
    // std::cout << "\tElapsed time (Resample Filter + Gaussian Interpolator): " << diff.count() << " s" << std::endl;

    // Filters to evaluate absolute difference
    const AbsoluteValueDifferenceImageFilter_3D::Pointer absDiffFilter_3D = AbsoluteValueDifferenceImageFilter_3D::New();
    const StatisticsImageFilterType_3D::Pointer statisticsImageFilter_3D = StatisticsImageFilterType_3D::New();

    // Compute LHS (Ax,y)
    absDiffFilter_3D->SetInput1( filter_OrientedGaussian_3D->GetOutput() );
    absDiffFilter_3D->SetInput2( filter_Resample_3D->GetOutput() );
    absDiffFilter_3D->Update();

    statisticsImageFilter_3D->SetInput( absDiffFilter_3D->GetOutput() );
    statisticsImageFilter_3D->Update();
    const double abs_diff = statisticsImageFilter_3D->GetSum();

    // MyITKImageHelper::showImage(filter_OrientedGaussian_3D->GetOutput(), filter_Resample_3D->GetOutput(), "GaussianFilter_InterpolateFilter");

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}


TEST_CASE( "Check GradientEuler3DTransformImageFilter", "[GradientEuler3DTransformImageFilter]"){
    
    // Define input and output
    const std::string dir_input = "../exampleData/";

    const std::string filename_slice = "FetalBrain_stack2_registered_midslice.nii.gz";

    const double tolerance = 1e-6;

    // Read images
    const ImageType3D::Pointer slice = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_slice);

    // Define Euler transform
    EulerTransformType::Pointer transform = EulerTransformType::New();

    // Gradient Euler transform
    const FilterType_GradientEuler_3D::Pointer filter_GradientEuler_3D = FilterType_GradientEuler_3D::New();
    filter_GradientEuler_3D->SetTransform(transform);
    filter_GradientEuler_3D->SetInput(slice);
    filter_GradientEuler_3D->Update();

    // Region and index to iterate over
    ImageType3D::IndexType index;
    itk::Point<PixelType, 3> point;
    ImageType3D::RegionType region = slice->GetBufferedRegion();
    itk::ImageRegionConstIteratorWithIndex<ImageType3D> it( slice, region );

    // Variables for comparison
    typedef itk::CovariantVector< PixelType, 18> CovariantVectorType;
    typedef itk::Image< CovariantVectorType, 3 > ImageCovariantVectorType;

    CovariantVectorType               jacobian;
    EulerTransformType::JacobianType  jacobian2;
    ImageCovariantVectorType::Pointer jacobian_slice = filter_GradientEuler_3D->GetOutput();

    // Walk the  region
    it.GoToBegin();

    while ( !it.IsAtEnd() ) {

        index = it.GetIndex();
        
        jacobian = jacobian_slice->GetPixel(index);
        // std::cout << jacobian << std::endl;
        
        slice->TransformIndexToPhysicalPoint(index, point);
        transform->ComputeJacobianWithRespectToParameters(point, jacobian2);
        // std::cout << jacobian2 << std::endl;

        // Check difference
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 6; ++j) {

                double abs_diff = std::abs(jacobian[i*6 + j] - jacobian2.GetElement(i,j));

                CHECK( abs_diff == Approx(0).epsilon(tolerance));
            }
        }

        ++it;
    }

    // Use different parameter
    EulerTransformType::ParametersType parameters(transform->GetNumberOfParameters());
    parameters[0] = 0.2;
    parameters[1] = 0.1;
    parameters[2] = 0.15;
    parameters[3] = -3.5;
    parameters[4] = 4.1;
    parameters[5] = 8.3;
    transform->SetParameters(parameters);
    filter_GradientEuler_3D->Update();
    // jacobian_slice = filter_GradientEuler_3D->GetOutput();

    it.GoToBegin();

    while ( !it.IsAtEnd() ) {

        index = it.GetIndex();
        
        jacobian = jacobian_slice->GetPixel(index);
        // std::cout << jacobian << std::endl;
        
        slice->TransformIndexToPhysicalPoint(index, point);
        transform->ComputeJacobianWithRespectToParameters(point, jacobian2);

        // Check difference
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 6; ++j) {

                double abs_diff = std::abs(jacobian[i*6 + j] - jacobian2.GetElement(i,j));

                CHECK( abs_diff == Approx(0).epsilon(tolerance));
            }
        }

        ++it;
    }


}
*/
TEST_CASE( "Check 3D itkOrientedGaussianInterpolateImageFilter Jacobian:", "[OrientedGaussian 3D Jacobian]") {

    // Define input and output
    const std::string dir_input = "../exampleData/";

    // const std::string filename_HR_volume = "FetalBrain_reconstruction_4stacks.nii.gz";
    const std::string filename_HR_volume = "FetalBrain_reconstruction_3stacks_myAlg.nii.gz";
    // const std::string filename_slice = "FetalBrain_stack2_registered_midslice.nii.gz";

    const double tolerance = 1e-6;

    // Set filter parameters
    const double alpha = 3;

    itk::Vector<double, 9> Cov_3D;
    Cov_3D.Fill(0);
    Cov_3D[0] = 0.26786367;
    Cov_3D[4] = 0.26786367;
    Cov_3D[8] = 2.67304559;

    // Read images
    // const ImageType3D::Pointer image = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_slice);
    const ImageType3D::Pointer image = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_HR_volume);

    // Variables for comparison
    typedef itk::CovariantVector< PixelType, 3> CovariantVectorType;
    typedef itk::Image< CovariantVectorType, 3 > ImageCovariantVectorType;

    // Oriented Gaussian Interpolate Image Filter
    const FilterType_OrientedGaussian_3D::Pointer filter_OrientedGaussian_3D = FilterType_OrientedGaussian_3D::New();

    filter_OrientedGaussian_3D->SetInput(image);
    filter_OrientedGaussian_3D->SetOutputParametersFromImage(image);
    filter_OrientedGaussian_3D->SetAlpha(alpha);
    // filter_OrientedGaussian_3D->SetCovariance(Cov_3D);
    filter_OrientedGaussian_3D->SetDefaultPixelValue( 0.0 );
    filter_OrientedGaussian_3D->SetUseJacobian(true);
    filter_OrientedGaussian_3D->Update();

    ImageType3D::Pointer image_filtered = filter_OrientedGaussian_3D->GetOutput();
    ImageCovariantVectorType::Pointer jacobian = filter_OrientedGaussian_3D->GetJacobian();

    // Gradient Image Filter
    FilterType_Gradient_3D::Pointer filter_Gradient_3D = FilterType_Gradient_3D::New();
    filter_Gradient_3D->SetInput(image_filtered);
    filter_Gradient_3D->SetUseImageSpacing(true);
    filter_Gradient_3D->SetUseImageDirection(true);
    filter_Gradient_3D->Update();
    ImageCovariantVectorType::Pointer jacobian2 = filter_Gradient_3D->GetOutput();

    // itk::GradientRecursiveGaussianImageFilter<ImageType3D>::Pointer filter_GradientGauss_3D = itk::GradientRecursiveGaussianImageFilter<ImageType3D>::New();
    // filter_GradientGauss_3D->SetInput(image_filtered);
    // filter_GradientGauss_3D->Update();
    // ImageCovariantVectorType::Pointer jacobian2 = filter_GradientGauss_3D->GetOutput();

    // Region and index to iterate over
    ImageType3D::IndexType index;
    itk::Point<PixelType, 3> point;
    ImageType3D::RegionType region = image->GetBufferedRegion();
    itk::ImageRegionConstIteratorWithIndex<ImageType3D> it( image, region );

    // Walk the  region
    it.GoToBegin();

    double abs_diff_total = 0.0;
    while ( !it.IsAtEnd() ) {

        index = it.GetIndex();
        CovariantVectorType jacobian_vector = jacobian->GetPixel(index);
        CovariantVectorType jacobian2_vector = jacobian2->GetPixel(index);

        // Check difference
        double abs_diff = 0.0;
        for (int i = 0; i < 3; ++i) {

            abs_diff += std::abs(jacobian_vector[i]-jacobian2_vector[i]);
            // std::cout << abs_diff << " ";
            // CHECK( abs_diff == Approx(0).epsilon(tolerance));
        }
        // std::cout << std::endl;
        abs_diff_total += abs_diff;
        ++it;
    }
    std::cout << "abs_diff_total = " << abs_diff_total << std::endl;

    MyITKImageHelper::showImage(jacobian, "JacobianGaussianFilter");
    MyITKImageHelper::showImage(jacobian2, "GradientFilter");

    // std::vector<ImageType3D::Pointer> derivative;
    // FilterType_Derivative_3D::Pointer filter_DerivativeFilter_3D = FilterType_Derivative_3D::New();
    // filter_DerivativeFilter_3D->SetInput(image_filtered);
    // filter_DerivativeFilter_3D->SetUseImageSpacing(true);
    // for (int i = 0; i < 3; ++i)
    // {
    //     filter_DerivativeFilter_3D->SetDirection(i);
    //     filter_DerivativeFilter_3D->Update();
    //     ImageType3D::Pointer out = filter_DerivativeFilter_3D->GetOutput();
    //     out->DisconnectPipeline();
    //     derivative.push_back(out);
    // }
    // MyITKImageHelper::showImage(derivative,"derivative");

}
/*
TEST_CASE( "Check 3D itkOrientedGaussianInterpolateImageFilter Jacobian: Real scenario", 
  "[OrientedGaussian 3D Jacobian: Real scenario]") {

    // Define input and output
    const std::string dir_input = "../exampleData/";

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

    // Oriented Gaussian Interpolate Image Filter
    // Measure time: Start
    auto start = std::chrono::system_clock::now();

    const FilterType_OrientedGaussian_3D::Pointer filter_OrientedGaussian_3D = FilterType_OrientedGaussian_3D::New();

    filter_OrientedGaussian_3D->SetInput(HR_volume);
    filter_OrientedGaussian_3D->SetOutputParametersFromImage(slice);
    filter_OrientedGaussian_3D->SetAlpha(alpha);
    filter_OrientedGaussian_3D->SetCovariance(Cov_3D);
    filter_OrientedGaussian_3D->SetDefaultPixelValue( 0.0 );
    filter_OrientedGaussian_3D->SetUseJacobian(true);
    filter_OrientedGaussian_3D->Update();

    // Measure time: Stop
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    const ImageType3D::RegionType::SizeType size = HR_volume->GetBufferedRegion().GetSize();
    std::cout << "Oriented Gaussian Filter test: 3D Realistic Images" << std::endl;
    std::cout << "\tSize of HR volume: " << size << std::endl;
    std::cout << "\tNumber of pixels HR volume: " << size[0]*size[1]*size[2] << std::endl;
    std::cout << "\tElapsed time (Gaussian Filter): " << diff.count() << " s" << std::endl;

    // std::cout << "UseJacobian = " << filter_OrientedGaussian_3D->GetUseJacobian() << std::endl;

    typedef FilterType_OrientedGaussian_3D::JacobianBaseType JacobianBaseType;
    FilterType_OrientedGaussian_3D::JacobianBaseType::Pointer jacobian = filter_OrientedGaussian_3D->GetJacobian();

    FilterType_GradientEuler_3D::Pointer filter_GradientEuler_3D = FilterType_GradientEuler_3D::New();
    filter_GradientEuler_3D->SetInput(slice);

    EulerTransformType::Pointer transform = EulerTransformType::New();
    EulerTransformType::ParametersType parameters(transform->GetNumberOfParameters());

    parameters[0] = 0.2;
    parameters[1] = 0.1;
    parameters[2] = 0.15;
    parameters[3] = -3.5;
    parameters[4] = 4.1;
    parameters[5] = 8.3;

    transform->SetParameters(parameters);
    filter_GradientEuler_3D->SetTransform(transform);
    // EulerTransformType::ConstPointer foo = filter_GradientEuler_3D->GetTransform();
    // std::cout << foo;
    filter_GradientEuler_3D->Update();

    FilterType_GradientEuler_3D::OutputImageType::Pointer foo = filter_GradientEuler_3D->GetOutput();
    // std::cout << foo;

    // FilterType_GradientEuler_3D::OutputImageRegionType region = foo->GetBufferedRegion();
    // std::cout << region;

    // typedef itk::VectorIndexSelectionCastImageFilter< JacobianBaseType, ImageType3D > IndexSelectionType;
    // IndexSelectionType::Pointer indexSelectionFilter = IndexSelectionType::New();
    // indexSelectionFilter->SetInput( filter_OrientedGaussian_3D->GetJacobian() );

    // std::vector<ImageType3D::Pointer> jacobian;
    // for (int i = 0; i < 3; ++i)
    // {
    //     indexSelectionFilter->SetIndex(i);
    //     indexSelectionFilter->Update();
    //     ImageType3D::Pointer tmp = indexSelectionFilter->GetOutput();
    //     tmp->DisconnectPipeline();
    //     jacobian.push_back(tmp);
    // }
    // MyITKImageHelper::showImage(jacobian, "Jacobian");


    // ImageType3D::Pointer test = indexSelectionFilter->GetOutput();
    // std::cout << test << std::endl;


    FilterType_OrientedGaussian_3D::IndexType index;
    index[0] = 40;
    index[1] = 45;
    index[2] = 0;
    // std::cout << v << std::endl;
    // std::cout << v->GetPixel(index) << std::endl;


    const std::string filename_image_2D = "2D_BrainWeb.nii.gz";
    const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

    FilterType_Derivative_2D::Pointer filter_DerivativeFilter_2D = FilterType_Derivative_2D::New();
    filter_DerivativeFilter_2D->SetInput(image_2D);
    filter_DerivativeFilter_2D->SetDirection(0);
    filter_DerivativeFilter_2D->Update();
    // std::cout << filter_DerivativeFilter_2D->GetOutput();
    // FilterType_Derivative_2D::OutputImageType::Pointer grad = filter_DerivativeFilter_2D->GetOutput();



    EulerTransformType::Pointer rigidTransform = EulerTransformType::New();
    rigidTransform->SetParameters(parameters);
    // std::cout << rigidTransform << std::endl;

    EulerTransformType::JacobianType v;

    PointType3D point;
    point[0] = 1;
    point[1] = 2;
    point[3] = -5;

    rigidTransform->ComputeJacobianWithRespectToParameters(point, v);
    // std::cout << v << std::endl;

    CHECK( 1 == Approx(0).epsilon(tolerance));
}


// TEST_CASE( "Malformed command line", "[Command line]" ) {

//     REQUIRE( system("../../bin/conwaysGameOfLife --help") == EXIT_SUCCESS );

//     ///**Would like to ask for '== EXIT_FAILURE' but return value
//     ///**system() in Mac gives 255 back whereas EXIT_FAILURE = 1 (wtf
//     REQUIRE( system("../../bin/conwaysGameOfLife") != EXIT_SUCCESS );

//     ///**Valid command line (therefore delete a possibly existing out.t
//     system("rm ../exampleData/out*.txt");
//     REQUIRE( system("../../bin/conwaysGameOfLife \
//         --i '../exampleData/InitialBoardRandom_10Times10.txt' \
//         --o '../exampleData/out.txt' \
//         --s 1 \
//         ") == EXIT_SUCCESS );

//     ///**output file now already given and shall not be overwritt
//     REQUIRE( system("../../bin/conwaysGameOfLife \
//         --i '../exampleData/InitialBoardRandom_10Times10.txt' \
//         --o '../exampleData/out.txt' \
//         --s 1 \
//         ") != EXIT_SUCCESS );

//     system("rm ../exampleData/out*.txt");
//     REQUIRE( system("../../bin/conwaysGameOfLife \
//         --i '../exampleData/InitialBoardRandom_10Times10.txt' \
//         --o '../exampleData/out.txt' \
//         --s 10000 \
//         ") != EXIT_SUCCESS );

//     system("rm ../exampleData/out*.txt");
//     REQUIRE( system("../../bin/conwaysGameOfLife \
//         --i '../exampleData/InitialBoardRandom_10Times10.txt' \
//         --s 10 \
//         ") != EXIT_SUCCESS );

//     system("rm ../exampleData/out*.txt");
//     REQUIRE( system("../../bin/conwaysGameOfLife \
//         --o '../exampleData/out.txt' \
//         --s 10000 \
//         ") != EXIT_SUCCESS );

//     system("rm ../exampleData/out*.txt");
//     REQUIRE( system("../../bin/conwaysGameOfLife \
//         --i '../exampleData/InitialBoardRandom_10Times10.txt' \
//         --o '../exampleData/out.txt' \
//         ") != EXIT_SUCCESS );

//     system("rm ../exampleData/out*.txt");
// }

// TEST_CASE( "Check whether file for input read is given", 
//   "[Not existing txt-file]") {
//     std::string sdir = "../exampleData/";

//     REQUIRE_THROWS_AS(Game *game = new Game(sdir+"nofile.txt"), MyException);
// }

*/


