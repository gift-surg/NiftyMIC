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

// My includes
#include "MyITKImageHelper.h"
#include "itkAdjointOrientedGaussianInterpolateImageFilter.h"
#include "itkOrientedGaussianInterpolateImageFunction.h"
// #include "MyException.h"


// Typedefs
typedef itk::ResampleImageFilter< ImageType2D, ImageType2D >  FilterType_Resample_2D;
typedef itk::ResampleImageFilter< ImageType3D, ImageType3D >  FilterType_Resample_3D;

typedef itk::AdjointOrientedGaussianInterpolateImageFilter<ImageType2D,ImageType2D>  FilterType_AdjointOrientedGaussian_2D;
typedef itk::AdjointOrientedGaussianInterpolateImageFilter<ImageType3D,ImageType3D>  FilterType_AdjointOrientedGaussian_3D;

typedef itk::OrientedGaussianInterpolateImageFunction< ImageType2D, PixelType >  InterpolatorType_2D;
typedef itk::OrientedGaussianInterpolateImageFunction< ImageType3D, PixelType >  InterpolatorType_3D;

typedef itk::MultiplyImageFilter< ImageType2D, ImageType2D, ImageType2D> MultiplyImageFilter_2D;
typedef itk::MultiplyImageFilter< ImageType3D, ImageType3D, ImageType3D> MultiplyImageFilter_3D;

typedef itk::AbsoluteValueDifferenceImageFilter< ImageType2D, ImageType2D, ImageType2D> AbsoluteValueDifferenceImageFilter_2D;
typedef itk::AbsoluteValueDifferenceImageFilter< ImageType3D, ImageType3D, ImageType3D> AbsoluteValueDifferenceImageFilter_3D;

typedef itk::StatisticsImageFilter<ImageType2D> StatisticsImageFilterType_2D;
typedef itk::StatisticsImageFilter<ImageType3D> StatisticsImageFilterType_3D;


// Unit tests
TEST_CASE( "Check 2D itkAdjointOrientedGaussianInterpolateImageFilter: Cross", 
  "[AdjointOrientedGaussian 2D: Cross]") {

    /* Define input and output */
    std::string dir_input = "../exampleData/";

    // const std::string filename_image_2D = "2D_SingleDot_50.nii.gz";
    const std::string filename_image_2D = "2D_Cross_50.nii.gz";
    // const std::string filename_image_2D = "2D_Text.nii.gz";
    // const std::string filename_image_2D = "2D_Lena_512.nii.gz";
    // const std::string filename_image_2D = "2D_SheppLoganPhantom_512.nii.gz";
    // const std::string filename_image_2D = "BrainWeb_2D.nii.gz";

    const double tolerance = 1e-6;

    /* Set filter parameters */
    const double alpha = 1;
    
    itk::Vector<double, 2> Sigma_2D;
    Sigma_2D[0] = 3;
    Sigma_2D[1] = 2;

    /* Read images */
    const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

    /* Adjoint Oriented Gaussian Interpolate Image Filter */
    const FilterType_AdjointOrientedGaussian_2D::Pointer filter_AdjointOrientedGaussian_2D = FilterType_AdjointOrientedGaussian_2D::New();

    filter_AdjointOrientedGaussian_2D->SetInput(image_2D);
    filter_AdjointOrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_AdjointOrientedGaussian_2D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_2D->SetSigma(Sigma_2D);
    filter_AdjointOrientedGaussian_2D->Update();

    /* Resample Image Filter */
    const InterpolatorType_2D::Pointer interpolator_2D = InterpolatorType_2D::New();
    interpolator_2D->SetSigma(Sigma_2D);
    interpolator_2D->SetAlpha(alpha);

    const  FilterType_Resample_2D::Pointer filter_Resample_2D = FilterType_Resample_2D::New();
    filter_Resample_2D->SetInput(image_2D);
    filter_Resample_2D->SetOutputParametersFromImage(image_2D);
    filter_Resample_2D->SetInterpolator(interpolator_2D);
    filter_Resample_2D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_2D->Update();  

    /* Filters to evaluate absolute difference */
    const MultiplyImageFilter_2D::Pointer multiplyFilter_2D = MultiplyImageFilter_2D::New();
    const AbsoluteValueDifferenceImageFilter_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilter_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilter_2D = StatisticsImageFilterType_2D::New();

    /* Compute LHS (Ax,y), x=y=image_2D */
    multiplyFilter_2D->SetInput1( filter_Resample_2D->GetOutput() );
    multiplyFilter_2D->SetInput2( image_2D );
    const ImageType2D::Pointer LHS = multiplyFilter_2D->GetOutput();

    statisticsImageFilter_2D->SetInput( LHS );
    statisticsImageFilter_2D->Update();
    const double sum_LHS = statisticsImageFilter_2D->GetSum();

    /* Compute RHS (x,A'y), x=y=image_2D */
    multiplyFilter_2D->SetInput1( image_2D );
    multiplyFilter_2D->SetInput2( filter_AdjointOrientedGaussian_2D->GetOutput() );
    const ImageType2D::Pointer RHS = multiplyFilter_2D->GetOutput();
    
    statisticsImageFilter_2D->SetInput( RHS );
    statisticsImageFilter_2D->Update();
    const double sum_RHS = statisticsImageFilter_2D->GetSum();

    /* compute | (Ax,y) - (x,A'y) | */
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // std::cout << "Filter: | (Ax,y) - (x,A'y) | = " << abs_diff << std::endl;
    // std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    // std::cout << "        (x,A'y) = " << sum_RHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

TEST_CASE( "Check 2D itkAdjointOrientedGaussianInterpolateImageFilter: Text", 
  "[AdjointOrientedGaussian 2D: Text]") {

    /* Define input and output */
    std::string dir_input = "../exampleData/";

    const std::string filename_image_2D = "2D_Text.nii.gz";

    const double tolerance = 1e-6;

    /* Set filter parameters */
    const double alpha = 1;
    
    itk::Vector<double, 2> Sigma_2D;
    Sigma_2D[0] = 3;
    Sigma_2D[1] = 2;

    /* Read images */
    const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

    /* Adjoint Oriented Gaussian Interpolate Image Filter */
    const FilterType_AdjointOrientedGaussian_2D::Pointer filter_AdjointOrientedGaussian_2D = FilterType_AdjointOrientedGaussian_2D::New();

    filter_AdjointOrientedGaussian_2D->SetInput(image_2D);
    filter_AdjointOrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_AdjointOrientedGaussian_2D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_2D->SetSigma(Sigma_2D);
    filter_AdjointOrientedGaussian_2D->Update();

    /* Resample Image Filter */
    const InterpolatorType_2D::Pointer interpolator_2D = InterpolatorType_2D::New();
    interpolator_2D->SetSigma(Sigma_2D);
    interpolator_2D->SetAlpha(alpha);

    const  FilterType_Resample_2D::Pointer filter_Resample_2D = FilterType_Resample_2D::New();
    filter_Resample_2D->SetInput(image_2D);
    filter_Resample_2D->SetOutputParametersFromImage(image_2D);
    filter_Resample_2D->SetInterpolator(interpolator_2D);
    filter_Resample_2D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_2D->Update();  

    /* Filters to evaluate absolute difference */
    const MultiplyImageFilter_2D::Pointer multiplyFilter_2D = MultiplyImageFilter_2D::New();
    const AbsoluteValueDifferenceImageFilter_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilter_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilter_2D = StatisticsImageFilterType_2D::New();

    /* Compute LHS (Ax,y), x=y=image_2D */
    multiplyFilter_2D->SetInput1( filter_Resample_2D->GetOutput() );
    multiplyFilter_2D->SetInput2( image_2D );
    const ImageType2D::Pointer LHS = multiplyFilter_2D->GetOutput();

    statisticsImageFilter_2D->SetInput( LHS );
    statisticsImageFilter_2D->Update();
    const double sum_LHS = statisticsImageFilter_2D->GetSum();

    /* Compute RHS (x,A'y), x=y=image_2D */
    multiplyFilter_2D->SetInput1( image_2D );
    multiplyFilter_2D->SetInput2( filter_AdjointOrientedGaussian_2D->GetOutput() );
    const ImageType2D::Pointer RHS = multiplyFilter_2D->GetOutput();
    
    statisticsImageFilter_2D->SetInput( RHS );
    statisticsImageFilter_2D->Update();
    const double sum_RHS = statisticsImageFilter_2D->GetSum();

    /* compute | (Ax,y) - (x,A'y) | */
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    std::cout << "Filter: | (Ax,y) - (x,A'y) | = " << abs_diff << std::endl;
    std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    std::cout << "        (x,A'y) = " << sum_RHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

TEST_CASE( "Check 2D itkAdjointOrientedGaussianInterpolateImageFilter: Shepp-Logan", 
  "[AdjointOrientedGaussian 2D: Shepp-Logan]") {

    /* Define input and output */
    std::string dir_input = "../exampleData/";

    const std::string filename_image_2D = "2D_SheppLoganPhantom_512.nii.gz";

    const double tolerance = 1e-6;

    /* Set filter parameters */
    const double alpha = 1;
    
    itk::Vector<double, 2> Sigma_2D;
    Sigma_2D[0] = 3;
    Sigma_2D[1] = 2;

    /* Read images */
    const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

    /* Adjoint Oriented Gaussian Interpolate Image Filter */
    const FilterType_AdjointOrientedGaussian_2D::Pointer filter_AdjointOrientedGaussian_2D = FilterType_AdjointOrientedGaussian_2D::New();

    filter_AdjointOrientedGaussian_2D->SetInput(image_2D);
    filter_AdjointOrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_AdjointOrientedGaussian_2D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_2D->SetSigma(Sigma_2D);
    filter_AdjointOrientedGaussian_2D->Update();

    /* Resample Image Filter */
    const InterpolatorType_2D::Pointer interpolator_2D = InterpolatorType_2D::New();
    interpolator_2D->SetSigma(Sigma_2D);
    interpolator_2D->SetAlpha(alpha);

    const  FilterType_Resample_2D::Pointer filter_Resample_2D = FilterType_Resample_2D::New();
    filter_Resample_2D->SetInput(image_2D);
    filter_Resample_2D->SetOutputParametersFromImage(image_2D);
    filter_Resample_2D->SetInterpolator(interpolator_2D);
    filter_Resample_2D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_2D->Update();  

    /* Filters to evaluate absolute difference */
    const MultiplyImageFilter_2D::Pointer multiplyFilter_2D = MultiplyImageFilter_2D::New();
    const AbsoluteValueDifferenceImageFilter_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilter_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilter_2D = StatisticsImageFilterType_2D::New();

    /* Compute LHS (Ax,y), x=y=image_2D */
    multiplyFilter_2D->SetInput1( filter_Resample_2D->GetOutput() );
    multiplyFilter_2D->SetInput2( image_2D );
    const ImageType2D::Pointer LHS = multiplyFilter_2D->GetOutput();

    statisticsImageFilter_2D->SetInput( LHS );
    statisticsImageFilter_2D->Update();
    const double sum_LHS = statisticsImageFilter_2D->GetSum();

    /* Compute RHS (x,A'y), x=y=image_2D */
    multiplyFilter_2D->SetInput1( image_2D );
    multiplyFilter_2D->SetInput2( filter_AdjointOrientedGaussian_2D->GetOutput() );
    const ImageType2D::Pointer RHS = multiplyFilter_2D->GetOutput();
    
    statisticsImageFilter_2D->SetInput( RHS );
    statisticsImageFilter_2D->Update();
    const double sum_RHS = statisticsImageFilter_2D->GetSum();

    /* compute | (Ax,y) - (x,A'y) | */
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // std::cout << "Filter: | (Ax,y) - (x,A'y) | = " << abs_diff << std::endl;
    // std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    // std::cout << "        (x,A'y) = " << sum_RHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}

TEST_CASE( "Check 2D itkAdjointOrientedGaussianInterpolateImageFilter: BrainWeb", 
  "[AdjointOrientedGaussian 2D: BrainWeb]") {

    /* Define input and output */
    std::string dir_input = "../exampleData/";

    const std::string filename_image_2D = "2D_BrainWeb.nii.gz";

    const double tolerance = 1e-6;

    /* Set filter parameters */
    const double alpha = 1;
    
    itk::Vector<double, 2> Sigma_2D;
    Sigma_2D[0] = 3;
    Sigma_2D[1] = 2;

    /* Read images */
    const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

    /* Adjoint Oriented Gaussian Interpolate Image Filter */
    const FilterType_AdjointOrientedGaussian_2D::Pointer filter_AdjointOrientedGaussian_2D = FilterType_AdjointOrientedGaussian_2D::New();

    filter_AdjointOrientedGaussian_2D->SetInput(image_2D);
    filter_AdjointOrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_AdjointOrientedGaussian_2D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_2D->SetSigma(Sigma_2D);
    filter_AdjointOrientedGaussian_2D->Update();

    /* Resample Image Filter */
    const InterpolatorType_2D::Pointer interpolator_2D = InterpolatorType_2D::New();
    interpolator_2D->SetSigma(Sigma_2D);
    interpolator_2D->SetAlpha(alpha);

    const  FilterType_Resample_2D::Pointer filter_Resample_2D = FilterType_Resample_2D::New();
    filter_Resample_2D->SetInput(image_2D);
    filter_Resample_2D->SetOutputParametersFromImage(image_2D);
    filter_Resample_2D->SetInterpolator(interpolator_2D);
    filter_Resample_2D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_2D->Update();  

    /* Filters to evaluate absolute difference */
    const MultiplyImageFilter_2D::Pointer multiplyFilter_2D = MultiplyImageFilter_2D::New();
    const AbsoluteValueDifferenceImageFilter_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilter_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilter_2D = StatisticsImageFilterType_2D::New();

    /* Compute LHS (Ax,y), x=y=image_2D */
    multiplyFilter_2D->SetInput1( filter_Resample_2D->GetOutput() );
    multiplyFilter_2D->SetInput2( image_2D );
    const ImageType2D::Pointer LHS = multiplyFilter_2D->GetOutput();

    statisticsImageFilter_2D->SetInput( LHS );
    statisticsImageFilter_2D->Update();
    const double sum_LHS = statisticsImageFilter_2D->GetSum();

    /* Compute RHS (x,A'y), x=y=image_2D */
    multiplyFilter_2D->SetInput1( image_2D );
    multiplyFilter_2D->SetInput2( filter_AdjointOrientedGaussian_2D->GetOutput() );
    const ImageType2D::Pointer RHS = multiplyFilter_2D->GetOutput();
    
    statisticsImageFilter_2D->SetInput( RHS );
    statisticsImageFilter_2D->Update();
    const double sum_RHS = statisticsImageFilter_2D->GetSum();

    /* compute | (Ax,y) - (x,A'y) | */
    const double abs_diff = std::abs(sum_LHS-sum_RHS);

    // MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput());

    std::cout << "Filter: | (Ax,y) - (x,A'y) | = " << abs_diff << std::endl;
    std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    std::cout << "        (x,A'y) = " << sum_RHS << std::endl;

    CHECK( abs_diff == Approx(0).epsilon(tolerance));
}



// TEST_CASE( "Malformed command line", "[Command line]" ) {

//     REQUIRE( system("../../bin/conwaysGameOfLife --help") == EXIT_SUCCESS );

//     //***Would like to ask for '== EXIT_FAILURE' but return value of
//     //***system() in Mac gives 255 back whereas EXIT_FAILURE = 1 (wtf!?)
//     REQUIRE( system("../../bin/conwaysGameOfLife") != EXIT_SUCCESS );

//     //***Valid command line (therefore delete a possibly existing out.txt)
//     system("rm ../exampleData/out*.txt");
//     REQUIRE( system("../../bin/conwaysGameOfLife \
//         --i '../exampleData/InitialBoardRandom_10Times10.txt' \
//         --o '../exampleData/out.txt' \
//         --s 1 \
//         ") == EXIT_SUCCESS );

//     //***output file now already given and shall not be overwritten:
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




