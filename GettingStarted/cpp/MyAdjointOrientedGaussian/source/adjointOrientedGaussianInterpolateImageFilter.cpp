/*! \brief Code to verify the implementation of itkAdjointOrientedGaussianInterpolateImageFilter.
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date February 2016
 */

#include <string>
#include <limits.h>     /* PATH_MAX */
#include <math.h>
#include <cstdlib>     /* system, NULL, EXIT_FAILURE */
#include <chrono>

#include <itkImage.h>
// #include <itkImageFileReader.h>
// #include <itkImageFileWriter.h>
// #include <itkNiftiImageIO.h>
#include <itkResampleImageFilter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkAbsoluteValueDifferenceImageFilter.h>
#include <itkStatisticsImageFilter.h>

// My includes
#include "MyITKImageHelper.h"
// #include "ImageFactory.h"
#include "itkAdjointOrientedGaussianInterpolateImageFilter.h"
#include "itkOrientedGaussianInterpolateImageFunction.h"


// Typedefs
typedef itk::ResampleImageFilter< ImageType2D, ImageType2D >  FilterType_OrientedGaussian_2D;
typedef itk::ResampleImageFilter< ImageType3D, ImageType3D >  FilterType_OrientedGaussian_3D;

typedef itk::AdjointOrientedGaussianInterpolateImageFilter<ImageType2D,ImageType2D>  FilterType_AdjointOrientedGaussian_2D;
typedef itk::AdjointOrientedGaussianInterpolateImageFilter<ImageType3D,ImageType3D>  FilterType_AdjointOrientedGaussian_3D;

typedef itk::OrientedGaussianInterpolateImageFunction< ImageType2D, PixelType >  OrientedGaussianInterpolatorType_2D;
typedef itk::OrientedGaussianInterpolateImageFunction< ImageType3D, PixelType >  OrientedGaussianInterpolatorType_3D;

typedef itk::MultiplyImageFilter< ImageType2D, ImageType2D, ImageType2D> MultiplyImageFilterType_2D;
typedef itk::MultiplyImageFilter< ImageType3D, ImageType3D, ImageType3D> MultiplyImageFilterType_3D;

typedef itk::SubtractImageFilter< ImageType2D, ImageType2D, ImageType2D> SubtractImageFilterType_2D;
typedef itk::SubtractImageFilter< ImageType3D, ImageType3D, ImageType3D> SubtractImageFilterType_3D;

typedef itk::AbsoluteValueDifferenceImageFilter< ImageType2D, ImageType2D, ImageType2D> AbsoluteValueDifferenceImageFilterType_2D;
typedef itk::AbsoluteValueDifferenceImageFilter< ImageType3D, ImageType3D, ImageType3D> AbsoluteValueDifferenceImageFilterType_3D;

typedef itk::StatisticsImageFilter<ImageType2D> StatisticsImageFilterType_2D;
typedef itk::StatisticsImageFilter<ImageType3D> StatisticsImageFilterType_3D;


 
int main(int, char*[])
{

  /* Define input and output */
  const std::string dir_input = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/data/";
  // const std::string dir_input = "../../../../data/";

  const std::string dir_output = "/tmp/";
  // const std::string dir_output = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/cpp/ITK_Examples/MyFunctions/results/";

  // const std::string filename_HR_volume = "BrainWeb_2D.png";
  const std::string filename_HR_volume = "FetalBrain_reconstruction_4stacks.nii.gz";
  const std::string filename_stack = "FetalBrain_stack2_registered.nii.gz";
  const std::string filename_slice = "FetalBrain_stack2_registered_midslice.nii.gz";
  const std::string filename_output = "test_output.nii.gz";

  // const std::string filename_image_2D = "2D_SingleDot_50.nii.gz";
  // const std::string filename_image_2D = "2D_Cross_50.nii.gz";

  // const std::string filename_image_2D = "2D_Text.nii.gz";

  const std::string filename_image_2D = "BrainWeb_2D.nii.gz";
  
  // const std::string filename_image_2D = "2D_Lena_512.nii.gz";
  // const std::string filename_image_2D = "2D_SheppLoganPhantom_512.nii.gz";
  
  // const std::string filename_image_3D = "3D_SingleDot_50.nii.gz";
  const std::string filename_image_3D = "3D_Cross_50.nii.gz";
  

  // MyITKImageHelper imageHelper;

  /* Set filter parameters */
  const double alpha = 2;
  
  itk::Vector<double, 2> Sigma_2D;
  Sigma_2D[0] = 3;
  Sigma_2D[1] = 2;

  itk::Vector<double, 9> Cov_3D;
  Cov_3D.Fill(0);
  Cov_3D[0] = 0.26786367;
  Cov_3D[4] = 0.26786367;
  Cov_3D[8] = 2.67304559;

  Cov_3D[0] = 2;
  Cov_3D[4] = 2;
  Cov_3D[8] = 3;
  

  /* Read images */
  // 2D
  const ImageType2D::Pointer image_2D = MyITKImageHelper::readImage<ImageType2D>(dir_input + filename_image_2D);

  // 3D
  const ImageType3D::Pointer HR_volume = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_HR_volume);
  const ImageType3D::Pointer stack = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_stack);
  const ImageType3D::Pointer slice = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_slice);

  const ImageType3D::Pointer image_3D = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename_image_3D);
  
  // MyITKImageHelper::showImage(image_2D);
  // MyITKImageHelper::showImage(image_3D);
  

  /////////////////////////////////////////////////////////////////////////////
  /* Test whether adjoint operator, i.e. |(Ax,y) - (x,A'y)| = 0            */
  /////////////////////////////////////////////////////////////////////////////

  // const bool choose_dimension_2D = true;
  const bool choose_dimension_2D = false;

  /* 2D: Check |(Ax,y) - (x,A'y)| = 0 */
  if (choose_dimension_2D) {

    // Measure time: Start
    auto start = std::chrono::system_clock::now();

    // Adjoint Oriented Gaussian Filter
    const FilterType_AdjointOrientedGaussian_2D::Pointer filter_AdjointOrientedGaussian_2D = FilterType_AdjointOrientedGaussian_2D::New();
    filter_AdjointOrientedGaussian_2D->SetInput(image_2D);
    filter_AdjointOrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_AdjointOrientedGaussian_2D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_2D->SetSigma(Sigma_2D);
    filter_AdjointOrientedGaussian_2D->SetDefaultPixelValue( 0.0 );  
    filter_AdjointOrientedGaussian_2D->Update();

    // Measure time: Stop
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "\t2D Elapsed time (Adjoint Operator): " << diff.count() << " s" << std::endl;


    // Measure time: Start
    start = std::chrono::system_clock::now();

    // Resample Image Filter with Oriented Gaussian Interpolation
    const OrientedGaussianInterpolatorType_2D::Pointer interpolator_OrientedGaussian_2D = OrientedGaussianInterpolatorType_2D::New();
    interpolator_OrientedGaussian_2D->SetSigma(Sigma_2D);
    interpolator_OrientedGaussian_2D->SetAlpha(alpha);

    const  FilterType_OrientedGaussian_2D::Pointer filter_OrientedGaussian_2D = FilterType_OrientedGaussian_2D::New();
    filter_OrientedGaussian_2D->SetInput(image_2D);
    filter_OrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
    filter_OrientedGaussian_2D->SetInterpolator(interpolator_OrientedGaussian_2D);
    filter_OrientedGaussian_2D->SetDefaultPixelValue( 0.0 );
    filter_OrientedGaussian_2D->Update();

    // Measure time: Stop
    end = std::chrono::system_clock::now();
    diff = end-start;
    std::cout << "\t2D Elapsed time (Forward Operator): " << diff.count() << " s" << std::endl;

    // Filters to evaluate absolute difference
    const MultiplyImageFilterType_2D::Pointer multiplyImageFilterType_2D = MultiplyImageFilterType_2D::New();
    const SubtractImageFilterType_2D::Pointer subtractImageFilterType_2D = SubtractImageFilterType_2D::New();
    const AbsoluteValueDifferenceImageFilterType_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilterType_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilterType_2D = StatisticsImageFilterType_2D::New();
    
    // Compute LHS Ax*y, i.e pointwise multiplication
    multiplyImageFilterType_2D->SetInput1( filter_OrientedGaussian_2D->GetOutput() );
    multiplyImageFilterType_2D->SetInput2( image_2D );
    multiplyImageFilterType_2D->Update();
    const ImageType2D::Pointer LHS = multiplyImageFilterType_2D->GetOutput();

    // Compute (Ax,y), i.e. get scalar value
    statisticsImageFilterType_2D->SetInput( LHS );
    statisticsImageFilterType_2D->Update();
    const double sum_LHS = statisticsImageFilterType_2D->GetSum();
    LHS->DisconnectPipeline(); // "I don't listen to what happens up my stream!"

    // Compute RHS x*A'y, i.e. pointwise multiplication
    multiplyImageFilterType_2D->SetInput1( image_2D );
    multiplyImageFilterType_2D->SetInput2( filter_AdjointOrientedGaussian_2D->GetOutput() );
    multiplyImageFilterType_2D->Update();
    const ImageType2D::Pointer RHS = multiplyImageFilterType_2D->GetOutput();

    // Compute (x,A'y), i.e. get scalar value
    statisticsImageFilterType_2D->SetInput( RHS );
    statisticsImageFilterType_2D->Update();
    const double sum_RHS = statisticsImageFilterType_2D->GetSum();
    RHS->DisconnectPipeline();

    std::cout << "Filter: |(Ax,y) - (x,A'y)| = " << std::abs(sum_LHS-sum_RHS) << std::endl;
    std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    std::cout << "        (x,A'y) = " << sum_RHS << std::endl;
    std::cout << "        |(Ax,y) - (x,A'y)|/(Ax,y) = " << std::abs(sum_LHS-sum_RHS)/sum_LHS << std::endl;


    // std::cout << "\n ------ ADJOINT ORIENTED GAUSSIAN ------ " << std::endl;
    // std::cout << "Covariance_2D = " << filter_AdjointOrientedGaussian_2D->GetCovariance() << std::endl;
    // std::cout << "Sigma_2D = " << filter_AdjointOrientedGaussian_2D->GetSigma() << std::endl;
    // std::cout << "Alpha_2D = " << filter_AdjointOrientedGaussian_2D->GetAlpha() << std::endl;

    // std::cout << " ------ ORIENTED GAUSSIAN ------ " << std::endl;
    // std::cout << "Covariance_2D = " << interpolator_OrientedGaussian_2D->GetCovariance() << std::endl;
    // std::cout << "Sigma_2D = " << interpolator_OrientedGaussian_2D->GetSigma() << std::endl;
    // std::cout << "Alpha_2D = " << interpolator_OrientedGaussian_2D->GetAlpha() << std::endl;


    // MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput(), filter_OrientedGaussian_2D->GetOutput(), "AdjointOrientedGaussian_OrientedGaussian");
    // MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput(), "AdjointOrientedGaussian");
    // MyITKImageHelper::showImage(filter_OrientedGaussian_2D->GetOutput(), "OrientedGaussian");
     // MyITKImageHelper::showImage(LHS, RHS, "LHS_RHS");

    // MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput(), "AdjointOrientedGaussian");
    // MyITKImageHelper::showImage(filter_OrientedGaussian_2D->GetOutput(), "OrientedGaussian");

    absDiffFilter_2D->SetInput1(LHS);
    absDiffFilter_2D->SetInput2(RHS);
    absDiffFilter_2D->Update();

    // MyITKImageHelper::showImage(absDiffFilter_2D->GetOutput(), "AbsoluteDifference");


    statisticsImageFilterType_2D->SetInput( absDiffFilter_2D->GetOutput() );
    statisticsImageFilterType_2D->Update();
    std::cout << "Mean: " << statisticsImageFilterType_2D->GetMean() << std::endl;
    std::cout << "Std.: " << statisticsImageFilterType_2D->GetSigma() << std::endl;
    std::cout << "Min: " << statisticsImageFilterType_2D->GetMinimum() << std::endl;
    std::cout << "Max: " << statisticsImageFilterType_2D->GetMaximum() << std::endl;
    std::cout << "Sum: " << statisticsImageFilterType_2D->GetSum() << std::endl;
     

    // MyITKImageHelper::showImage(statisticsImageFilterType_2D->GetOutput(), "statisticsImageFilterType_2D");
    // MyITKImageHelper::showImage(subtractImageFilterType_2D->GetOutput(), "subtractImageFilterType_2D");


    // MyITKImageHelper::showImage(image_2D);

    ImageType2D::Pointer res_forward_2D = filter_OrientedGaussian_2D->GetOutput();
    ImageType2D::Pointer res_backward_2D = filter_AdjointOrientedGaussian_2D->GetOutput();

    ImageType2D::RegionType region = image_2D->GetLargestPossibleRegion();

    itk::ImageRegionConstIteratorWithIndex<ImageType2D> It_2D(image_2D, region);

    // std::cout << " Region = " << region << std::endl;

    double sum_forward = 0.0;
    double sum_backward = 0.0;

    for( It_2D.GoToBegin(); !It_2D.IsAtEnd(); ++It_2D )
      {
      sum_backward  += res_backward_2D->GetPixel(It_2D.GetIndex()) * It_2D.Get();
      sum_forward   += It_2D.Get() * res_forward_2D->GetPixel(It_2D.GetIndex());
      }

    std::cout << "\nBy hand: |(Ax,y) - (x,A'y)| = " << std::abs(sum_forward-sum_backward) << std::endl;
    std::cout << "        (Ax,y) = " << sum_forward << std::endl;
    std::cout << "        (x,A'y) = " << sum_backward << std::endl;

  }

  /* 3D: Check |(Ax,y) - (x,A'y)| = 0 */
  // x = HR volume
  // y = LR slice
  else {

    // Measure time: Start
    auto start = std::chrono::system_clock::now();

    // Adjoint Oriented Gaussian Filter
    const FilterType_AdjointOrientedGaussian_3D::Pointer filter_AdjointOrientedGaussian_3D = FilterType_AdjointOrientedGaussian_3D::New();

    // filter_AdjointOrientedGaussian_3D->SetOutputParametersFromImage(image_3D);
    // filter_AdjointOrientedGaussian_3D->SetInput(image_3D);
    
    filter_AdjointOrientedGaussian_3D->SetInput(slice);
    filter_AdjointOrientedGaussian_3D->SetOutputParametersFromImage(HR_volume);

    filter_AdjointOrientedGaussian_3D->SetAlpha(alpha);
    filter_AdjointOrientedGaussian_3D->SetCovariance(Cov_3D);
    filter_AdjointOrientedGaussian_3D->SetDefaultPixelValue( 0.0 );  
    filter_AdjointOrientedGaussian_3D->Update();

    // Measure time: Stop
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "\t3D Elapsed time (Adjoint Operator): " << diff.count() << " s" << std::endl;


    // Measure time: Start
    start = std::chrono::system_clock::now();

    // Resample Image Filter with Oriented Gaussian Interpolation
    const OrientedGaussianInterpolatorType_3D::Pointer interpolator_OrientedGaussian_3D = OrientedGaussianInterpolatorType_3D::New();
    interpolator_OrientedGaussian_3D->SetCovariance(Cov_3D);
    interpolator_OrientedGaussian_3D->SetAlpha(alpha);

    const  FilterType_OrientedGaussian_3D::Pointer filter_OrientedGaussian_3D = FilterType_OrientedGaussian_3D::New();
    filter_OrientedGaussian_3D->SetInput(HR_volume);
    filter_OrientedGaussian_3D->SetOutputParametersFromImage(slice);
    filter_OrientedGaussian_3D->SetInterpolator(interpolator_OrientedGaussian_3D);
    filter_OrientedGaussian_3D->SetDefaultPixelValue( 0.0 );  
    filter_OrientedGaussian_3D->Update();  

    // Measure time: Stop
    end = std::chrono::system_clock::now();
    diff = end-start;
    std::cout << "\t3D Elapsed time (Forward Operator): " << diff.count() << " s" << std::endl;

    // Filters to evaluate absolute difference
    const MultiplyImageFilterType_3D::Pointer multiplyImageFilterType_3D = MultiplyImageFilterType_3D::New();
    const AbsoluteValueDifferenceImageFilterType_3D::Pointer absDiffFilter_3D = AbsoluteValueDifferenceImageFilterType_3D::New();
    const StatisticsImageFilterType_3D::Pointer statisticsImageFilterType_3D = StatisticsImageFilterType_3D::New();
    
    // Compute LHS
    multiplyImageFilterType_3D->SetInput1( filter_OrientedGaussian_3D->GetOutput() );
    multiplyImageFilterType_3D->SetInput2( slice );
    const ImageType3D::Pointer LHS = multiplyImageFilterType_3D->GetOutput();

    // TODO: Makes a difference whether it is computed here or after "Compute RHS"!?
    statisticsImageFilterType_3D->SetInput( LHS );
    statisticsImageFilterType_3D->Update();
    const double sum_LHS = statisticsImageFilterType_3D->GetSum();

    // Compute RHS
    multiplyImageFilterType_3D->SetInput1( HR_volume );
    multiplyImageFilterType_3D->SetInput2( filter_AdjointOrientedGaussian_3D->GetOutput() );
    const ImageType3D::Pointer RHS = multiplyImageFilterType_3D->GetOutput();

    // TODO: Makes a difference whether it is computed here or before "Compute RHS"!?
    // statisticsImageFilterType_3D->SetInput( LHS );
    // statisticsImageFilterType_3D->Update();
    // const double sum_LHS = statisticsImageFilterType_3D->GetSum();
    
    statisticsImageFilterType_3D->SetInput( RHS );
    statisticsImageFilterType_3D->Update();
    const double sum_RHS = statisticsImageFilterType_3D->GetSum();

    std::cout << "Filter: |(Ax,y) - (x,A'y)| = " << std::abs(sum_LHS-sum_RHS) << std::endl;
    std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    std::cout << "        (x,A'y) = " << sum_RHS << std::endl;
    std::cout << "        |(Ax,y) - (x,A'y)|/(Ax,y) = " << std::abs(sum_LHS-sum_RHS)/sum_LHS << std::endl;


    // MyITKImageHelper::showImage(LHS, "LHS");
    // MyITKImageHelper::showImage(RHS, "RHS");
    
    // MyITKImageHelper::showImage(image_3D);

  }


  return EXIT_SUCCESS;
}
 