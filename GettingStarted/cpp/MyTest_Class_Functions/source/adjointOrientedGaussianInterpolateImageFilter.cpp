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
typedef itk::ResampleImageFilter< ImageType2D, ImageType2D >  FilterType_Resample_2D;
typedef itk::ResampleImageFilter< ImageType3D, ImageType3D >  FilterType_Resample_3D;

typedef itk::AdjointOrientedGaussianInterpolateImageFilter<ImageType2D,ImageType2D>  FilterType_AdjointOrientedGaussian_2D;
typedef itk::AdjointOrientedGaussianInterpolateImageFilter<ImageType3D,ImageType3D>  FilterType_AdjointOrientedGaussian_3D;

typedef itk::OrientedGaussianInterpolateImageFunction< ImageType2D, PixelType >  InterpolatorType_2D;
typedef itk::OrientedGaussianInterpolateImageFunction< ImageType3D, PixelType >  InterpolatorType_3D;

typedef itk::MultiplyImageFilter< ImageType2D, ImageType2D, ImageType2D> MultiplyImageFilter_2D;
typedef itk::MultiplyImageFilter< ImageType3D, ImageType3D, ImageType3D> MultiplyImageFilter_3D;

typedef itk::SubtractImageFilter< ImageType2D, ImageType2D, ImageType2D> SubtractImageFilter_2D;
typedef itk::SubtractImageFilter< ImageType3D, ImageType3D, ImageType3D> SubtractImageFilter_3D;

typedef itk::AbsoluteValueDifferenceImageFilter< ImageType2D, ImageType2D, ImageType2D> AbsoluteValueDifferenceImageFilter_2D;
typedef itk::AbsoluteValueDifferenceImageFilter< ImageType3D, ImageType3D, ImageType3D> AbsoluteValueDifferenceImageFilter_3D;

typedef itk::StatisticsImageFilter<ImageType2D> StatisticsImageFilterType_2D;
typedef itk::StatisticsImageFilter<ImageType3D> StatisticsImageFilterType_3D;


 
int main(int, char*[])
{

  /* Define input and output */
  // const std::string dir_input = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/data/";
  const std::string dir_input = "../../../../data/";
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
  // const std::string filename_image_2D = "2D_Text_double.nii.gz";
  // const std::string filename_image_2D = "2D_Text_float32.nii.gz";
  // const std::string filename_image_2D = "2D_Text_float64.nii.gz";

  const std::string filename_image_2D = "BrainWeb_2D.nii.gz";
  // const std::string filename_image_2D = "BrainWeb_2D_float64.nii.gz";
  
  // const std::string filename_image_2D = "2D_Lena_512.nii.gz";
  // const std::string filename_image_2D = "2D_SheppLoganPhantom_512.nii.gz";
  
  const std::string filename_image_3D = "3D_SingleDot_50.nii.gz";
  

  // MyITKImageHelper imageHelper;

  /* Set filter parameters */
  const double alpha = 2;
  
  itk::Vector<double, 2> Sigma_2D;
  itk::Vector<double, 3> Sigma_3D;

  Sigma_2D[0] = 3;
  Sigma_2D[1] = 2;

  Sigma_3D[0] = 3;
  Sigma_3D[1] = 2;
  Sigma_3D[2] = 1;

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

  /* Adjoint Oriented Gaussian Interpolate Image Filter */
  // 2D
  const FilterType_AdjointOrientedGaussian_2D::Pointer filter_AdjointOrientedGaussian_2D = FilterType_AdjointOrientedGaussian_2D::New();

  // 3D
  const FilterType_AdjointOrientedGaussian_3D::Pointer filter_AdjointOrientedGaussian_3D = FilterType_AdjointOrientedGaussian_3D::New();
 
  /* Parametrize filter */
  // 2D
  filter_AdjointOrientedGaussian_2D->SetInput(image_2D);
  filter_AdjointOrientedGaussian_2D->SetOutputParametersFromImage(image_2D);
  filter_AdjointOrientedGaussian_2D->SetAlpha(alpha);
  filter_AdjointOrientedGaussian_2D->SetSigma(Sigma_2D);
  filter_AdjointOrientedGaussian_2D->Update();

  // 3D
  // filter_AdjointOrientedGaussian_3D->SetInput(image_3D);
  // filter_AdjointOrientedGaussian_3D->SetOutputParametersFromImage(image_3D);

  filter_AdjointOrientedGaussian_3D->SetInput(slice);
  filter_AdjointOrientedGaussian_3D->SetOutputParametersFromImage(HR_volume);

  filter_AdjointOrientedGaussian_3D->SetAlpha(alpha);
  filter_AdjointOrientedGaussian_3D->SetSigma(Sigma_3D);
  filter_AdjointOrientedGaussian_3D->Update();
  
  // std::cout << "Covariance_3D = " << filter_AdjointOrientedGaussian_3D->GetCovariance() << std::endl;
  // std::cout << "Sigma_3D = " << filter_AdjointOrientedGaussian_3D->GetSigma() << std::endl;
  // std::cout << "Alpha_3D = " << filter_AdjointOrientedGaussian_3D->GetAlpha() << std::endl;

  // std::cout << " ------ RESAMPLE IMAGE FILTER ------ " << std::endl;
  // std::cout << filter_Resample_2D << std::endl;

  // std::cout << " ------ ADJOINT ORIENTED GAUSSIAN IMAGE FILTER ------ " << std::endl;
  // std::cout << filter_AdjointOrientedGaussian_2D << std::endl;
 

  /////////////////////////////////////////////////////////////////////////////
  /* Test whether adjoint operator, i.e. | (Ax,y) - (x,A'y) | = 0            */
  /////////////////////////////////////////////////////////////////////////////

  const bool choose_dimension_2D = true;
  // const bool choose_dimension_2D = false;

  /* 2D: Check | (Ax,y) - (x,A'y) | = 0 */
  if (choose_dimension_2D) {

    // ResampleImageFilter
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
    const MultiplyImageFilter_2D::Pointer multiplyImageFilter_2D = MultiplyImageFilter_2D::New();
    const SubtractImageFilter_2D::Pointer subtractImageFilter_2D = SubtractImageFilter_2D::New();
    const AbsoluteValueDifferenceImageFilter_2D::Pointer absDiffFilter_2D = AbsoluteValueDifferenceImageFilter_2D::New();
    const StatisticsImageFilterType_2D::Pointer statisticsImageFilter_2D = StatisticsImageFilterType_2D::New();
    
    // Compute LHS
    multiplyImageFilter_2D->SetInput1( filter_Resample_2D->GetOutput() );
    multiplyImageFilter_2D->SetInput2( image_2D );
    multiplyImageFilter_2D->Update();
    const ImageType2D::Pointer LHS = multiplyImageFilter_2D->GetOutput();

    // TODO: Makes a difference whether it is computed here or after "Compute RHS"!?
    statisticsImageFilter_2D->SetInput( LHS );
    statisticsImageFilter_2D->Update();
    const double sum_LHS = statisticsImageFilter_2D->GetSum();
    statisticsImageFilter_2D->ResetPipeline();

    // Compute RHS
    multiplyImageFilter_2D->SetInput1( image_2D );
    multiplyImageFilter_2D->SetInput2( filter_AdjointOrientedGaussian_2D->GetOutput() );
    multiplyImageFilter_2D->Update();
    const ImageType2D::Pointer RHS = multiplyImageFilter_2D->GetOutput();

    // TODO: Makes a difference whether it is computed here or before "Compute RHS"!?
    // statisticsImageFilter_2D->SetInput( LHS );
    // statisticsImageFilter_2D->Update();
    // const double sum_LHS = statisticsImageFilter_2D->GetSum();
    // statisticsImageFilter_2D->ResetPipeline();

    
    statisticsImageFilter_2D->SetInput( RHS );
    statisticsImageFilter_2D->Update();
    const double sum_RHS = statisticsImageFilter_2D->GetSum();

    std::cout << "Filter: | (Ax,y) - (x,A'y) | = " << std::abs(sum_LHS-sum_RHS) << std::endl;
    std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    std::cout << "        (x,A'y) = " << sum_RHS << std::endl;


    std::cout << "\n ------ ADJOINT ORIENTED GAUSSIAN ------ " << std::endl;
    std::cout << "Covariance_2D = " << filter_AdjointOrientedGaussian_2D->GetCovariance() << std::endl;
    std::cout << "Sigma_2D = " << filter_AdjointOrientedGaussian_2D->GetSigma() << std::endl;
    std::cout << "Alpha_2D = " << filter_AdjointOrientedGaussian_2D->GetAlpha() << std::endl;

    std::cout << " ------ ORIENTED GAUSSIAN ------ " << std::endl;
    std::cout << "Covariance_2D = " << interpolator_2D->GetCovariance() << std::endl;
    std::cout << "Sigma_2D = " << interpolator_2D->GetSigma() << std::endl;
    std::cout << "Alpha_2D = " << interpolator_2D->GetAlpha() << std::endl;


    // MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput(), filter_AdjointOrientedGaussian_2D->GetOutput(), "AdjointOrientedGaussian_OrientedGaussian");
     // MyITKImageHelper::showImage(LHS, RHS, "LHS_RHS");

    MyITKImageHelper::showImage(filter_AdjointOrientedGaussian_2D->GetOutput(), "AdjointOrientedGaussian");
    MyITKImageHelper::showImage(filter_Resample_2D->GetOutput(), "OrientedGaussian");

    subtractImageFilter_2D->SetInput1(LHS);
    subtractImageFilter_2D->SetInput2(RHS);
    subtractImageFilter_2D->Update();

    statisticsImageFilter_2D->SetInput( subtractImageFilter_2D->GetOutput() );
    std::cout << "Mean: " << statisticsImageFilter_2D->GetMean() << std::endl;
    std::cout << "Std.: " << statisticsImageFilter_2D->GetSigma() << std::endl;
    std::cout << "Min: " << statisticsImageFilter_2D->GetMinimum() << std::endl;
    std::cout << "Max: " << statisticsImageFilter_2D->GetMaximum() << std::endl;
     

    // MyITKImageHelper::showImage(statisticsImageFilter_2D->GetOutput(), "statisticsImageFilter_2D");
    // MyITKImageHelper::showImage(subtractImageFilter_2D->GetOutput(), "subtractImageFilter_2D");


    // MyITKImageHelper::showImage(image_2D);

    ImageType2D::Pointer res_forward_2D = filter_Resample_2D->GetOutput();
    ImageType2D::Pointer res_backward_2D = filter_AdjointOrientedGaussian_2D->GetOutput();

    ImageType2D::RegionType region = image_2D->GetLargestPossibleRegion();

    itk::ImageRegionConstIteratorWithIndex<ImageType2D> It_2D(image_2D, region);

    std::cout << " Region = " << region << std::endl;

    double sum_forward = 0.0;
    double sum_backward = 0.0;

    for( It_2D.GoToBegin(); !It_2D.IsAtEnd(); ++It_2D )
      {
      sum_backward  += res_backward_2D->GetPixel(It_2D.GetIndex()) * It_2D.Get();
      sum_forward   += It_2D.Get() * res_forward_2D->GetPixel(It_2D.GetIndex());
      }

    std::cout << "By hand: | (Ax,y) - (x,A'y) | = " << std::abs(sum_forward-sum_backward) << std::endl;
    std::cout << "        (Ax,y) = " << sum_forward << std::endl;
    std::cout << "        (x,A'y) = " << sum_backward << std::endl;

  }

  /* 3D: Check | (Ax,y) - (x,A'y) | = 0 */
  // x = HR volume
  // y = LR slice
  else {
    // ResampleImageFilter
    const InterpolatorType_3D::Pointer interpolator_3D = InterpolatorType_3D::New();
    interpolator_3D->SetSigma(Sigma_3D);
    interpolator_3D->SetAlpha(alpha);

    const  FilterType_Resample_3D::Pointer filter_Resample_3D = FilterType_Resample_3D::New();
    filter_Resample_3D->SetInput(HR_volume);
    filter_Resample_3D->SetOutputParametersFromImage(slice);
    filter_Resample_3D->SetInterpolator(interpolator_3D);
    filter_Resample_3D->SetDefaultPixelValue( 0.0 );  
    filter_Resample_3D->Update();  

    // Filters to evaluate absolute difference
    const MultiplyImageFilter_3D::Pointer multiplyImageFilter_3D = MultiplyImageFilter_3D::New();
    const AbsoluteValueDifferenceImageFilter_3D::Pointer absDiffFilter_3D = AbsoluteValueDifferenceImageFilter_3D::New();
    const StatisticsImageFilterType_3D::Pointer statisticsImageFilter_3D = StatisticsImageFilterType_3D::New();
    
    // Compute LHS
    multiplyImageFilter_3D->SetInput1( filter_Resample_3D->GetOutput() );
    multiplyImageFilter_3D->SetInput2( slice );
    const ImageType3D::Pointer LHS = multiplyImageFilter_3D->GetOutput();

    // TODO: Makes a difference whether it is computed here or after "Compute RHS"!?
    statisticsImageFilter_3D->SetInput( LHS );
    statisticsImageFilter_3D->Update();
    const double sum_LHS = statisticsImageFilter_3D->GetSum();

    // Compute RHS
    multiplyImageFilter_3D->SetInput1( HR_volume );
    multiplyImageFilter_3D->SetInput2( filter_AdjointOrientedGaussian_3D->GetOutput() );
    const ImageType3D::Pointer RHS = multiplyImageFilter_3D->GetOutput();

    // TODO: Makes a difference whether it is computed here or before "Compute RHS"!?
    // statisticsImageFilter_3D->SetInput( LHS );
    // statisticsImageFilter_3D->Update();
    // const double sum_LHS = statisticsImageFilter_3D->GetSum();
    
    statisticsImageFilter_3D->SetInput( RHS );
    statisticsImageFilter_3D->Update();
    const double sum_RHS = statisticsImageFilter_3D->GetSum();

    std::cout << "Filter: | (Ax,y) - (x,A'y) | = " << std::abs(sum_LHS-sum_RHS) << std::endl;
    std::cout << "        (Ax,y) = " << sum_LHS << std::endl;
    std::cout << "        (x,A'y) = " << sum_RHS << std::endl;


    MyITKImageHelper::showImage(LHS, "LHS");
    MyITKImageHelper::showImage(RHS, "RHS");
    
    MyITKImageHelper::showImage(image_3D);

  }


  return EXIT_SUCCESS;
}
 