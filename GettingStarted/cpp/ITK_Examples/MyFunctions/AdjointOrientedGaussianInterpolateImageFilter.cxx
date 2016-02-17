/*! \brief Code to verify the implementation of itkAdjointOrientedGaussianInterpolateImageFilter.
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date February 2016
 */

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <itkNiftiImageIO.h>
#include "itkResampleImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkAdjointOrientedGaussianInterpolateImageFilter.h"
#include "itkOrientedGaussianInterpolateImageFunction.h"
 
#include <string>
#include <limits.h> /* PATH_MAX */
#include <math.h>
#include <stdlib.h>     /* system, NULL, EXIT_FAILURE */

template <typename ImageType>
void showImage(const ImageType * image){

  const std::string dir_output = "/tmp/";
  const std::string filename_output = "test.nii.gz";

  typename itk::ImageFileWriter< ImageType >::Pointer 
    writer = itk::ImageFileWriter< ImageType >::New();

  /* Write images */
  writer->SetFileName( dir_output + filename_output );
  writer->SetInput( image );
  writer->Update();

  /* View image via itksnap */
  std::string  cmd = "itksnap ";
  cmd += "-g " + dir_output + filename_output;
  cmd += "& ";

  system(cmd.c_str());

}

// template<>
// void showImage<OutputImageType>(OutputImageType image);
 
int main(int, char*[])
{
  /* Define input and output */
  const std::string dir_input = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/data/";
  const std::string dir_output = "/tmp/";
  // const std::string dir_output = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/cpp/ITK_Examples/MyFunctions/results/";

  // const std::string filename_HR_volume = "BrainWeb_2D.png";
  const std::string filename_HR_volume = "FetalBrain_reconstruction_4stacks.nii.gz";
  const std::string filename_stack = "FetalBrain_stack2_registered.nii.gz";
  const std::string filename_slice = "FetalBrain_stack2_registered_midslice.nii.gz";
  const std::string filename_output = "test_output.nii.gz";
  const std::string filename_SingleDot_2D = "SingleDot_2D_50.nii.gz";
  const std::string filename_SingleDot_3D = "SingleDot_3D_50.nii.gz";

  /* Define dimension of image and pixel types */
  const     unsigned int   Dimension = 3;
  typedef   double  InputPixelType;
  typedef   double  OutputPixelType;
  // typedef   unsigned char  InputPixelType;
  // typedef   unsigned char  OutputPixelType;

  /* Define types of input and outpyt image */
  typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
  typedef itk::Image< OutputPixelType, Dimension >   OutputImageType;
  typedef itk::Image< OutputPixelType, 2 >           ImageType_2D;

  /* Define types of reader and writer */
  typedef itk::ImageFileReader< InputImageType  >  ReaderType;
  typedef itk::ImageFileReader< ImageType_2D  >     ReaderType_2D;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  
  /* Instantiate reader and writer */
  ReaderType::Pointer reader_HR_volume = ReaderType::New();
  ReaderType::Pointer reader_stack = ReaderType::New();
  ReaderType::Pointer reader_slice = ReaderType::New();
  ReaderType_2D::Pointer reader_SingleDot_2D = ReaderType_2D::New();
  ReaderType::Pointer reader_SingleDot_3D= ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  /* Set image IO type to nifti */
  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
  reader_HR_volume->SetImageIO(imageIO);
  reader_stack->SetImageIO(imageIO);
  reader_slice->SetImageIO(imageIO);
  reader_SingleDot_2D->SetImageIO(imageIO);
  reader_SingleDot_3D->SetImageIO(imageIO);

  /* Read images */
  reader_HR_volume->SetFileName( dir_input + filename_HR_volume );
  reader_HR_volume->Update();

  reader_stack->SetFileName( dir_input + filename_stack );
  reader_stack->Update();

  reader_slice->SetFileName( dir_input + filename_slice );
  reader_slice->Update();

  reader_SingleDot_2D->SetFileName( dir_input + filename_SingleDot_2D );
  reader_SingleDot_2D->Update();

  reader_SingleDot_3D->SetFileName( dir_input + filename_SingleDot_3D );
  reader_SingleDot_3D->Update();

  /* Get images */
  const InputImageType* HR_volume = reader_HR_volume->GetOutput();
  const InputImageType* stack = reader_stack->GetOutput();
  const InputImageType* slice = reader_slice->GetOutput();
  const ImageType_2D* SingleDot_2D = reader_SingleDot_2D->GetOutput();
  const InputImageType* SingleDot_3D = reader_SingleDot_3D->GetOutput();

  /* Adjoint Oriented Gaussian Interpolae Image Filter */


  // typedef itk::AdjointOrientedGaussianInterpolateImageFilter<InputImageType,InputImageType>  FilterType_AdjointOrientedGaussian;
  typedef itk::AdjointOrientedGaussianInterpolateImageFilter<ImageType_2D,ImageType_2D>  FilterType_AdjointOrientedGaussian;

  FilterType_AdjointOrientedGaussian::Pointer filter_AdjointOrientedGaussian = FilterType_AdjointOrientedGaussian::New();
 
  /* Set filter */
  // filter_AdjointOrientedGaussian->SetInput(slice);
  // filter_AdjointOrientedGaussian->SetOutputParametersFromImage(HR_volume);

  filter_AdjointOrientedGaussian->SetInput(SingleDot_2D);
  filter_AdjointOrientedGaussian->SetOutputParametersFromImage(SingleDot_2D);

  const double alpha = 3;
  
  itk::Vector<double, 2> Sigma;
  // itk::Vector<double, 3> Sigma;
  Sigma[0] = 3;
  Sigma[1] = 2;
  // Sigma[2] = 1;


  filter_AdjointOrientedGaussian->SetSigma(Sigma);
  filter_AdjointOrientedGaussian->SetAlpha(alpha);

  filter_AdjointOrientedGaussian->Update();

  /* Resample Image Filter */
  typedef itk::ResampleImageFilter< InputImageType, OutputImageType >  FilterType_Resample;
  FilterType_Resample::Pointer filter_Resample = FilterType_Resample::New();
  filter_Resample->SetInput(slice);
  filter_Resample->SetOutputParametersFromImage(HR_volume);
  // filter_Resample->Update();

  std::cout << "Covariance = " << filter_AdjointOrientedGaussian->GetCovariance() << std::endl;
  std::cout << "Sigma = " << filter_AdjointOrientedGaussian->GetSigma() << std::endl;
  std::cout << "Alpha = " << filter_AdjointOrientedGaussian->GetAlpha() << std::endl;

 
  // std::cout << " ------ RESAMPLE IMAGE FILTER ------ " << std::endl;
  // std::cout << filter_Resample << std::endl;

  // std::cout << " ------ ADJOINT ORIENTED GAUSSIAN IMAGE FILTER ------ " << std::endl;
  // std::cout << filter_AdjointOrientedGaussian << std::endl;
 
  // showImage(filter_AdjointOrientedGaussian->GetOutput());
  // showImage(SingleDot_2D);



  /////////////////////////////////////////////////////////////////////////////
  /* Test whether adjoint operator does something meaningful */
  /////////////////////////////////////////////////////////////////////////////

  /* Resample Image Filter */
  // typedef itk::ResampleImageFilter< InputImageType, OutputImageType >  FilterType;
  typedef itk::ResampleImageFilter< ImageType_2D, ImageType_2D >  FilterType;
  FilterType::Pointer filter = FilterType::New();


  // typedef itk::OrientedGaussianInterpolateImageFunction< InputImageType, double >  InterpolatorType;
  typedef itk::OrientedGaussianInterpolateImageFunction< ImageType_2D, double >  InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  interpolator->SetSigma(Sigma);
  interpolator->SetAlpha(alpha);

  filter->SetInput(SingleDot_2D);
  filter->SetOutputParametersFromImage(SingleDot_2D);
  filter->SetInterpolator(interpolator);
  filter->SetDefaultPixelValue( 0 );  
  filter->Update();  


  ImageType_2D* res_forward_2D = filter->GetOutput();
  ImageType_2D* res_backward_2D = filter_AdjointOrientedGaussian->GetOutput();

  ImageType_2D::RegionType region = SingleDot_2D->GetLargestPossibleRegion();

  itk::ImageRegionConstIteratorWithIndex<ImageType_2D> It(SingleDot_2D, region);

  double sum_forward = 0.0;
  double sum_backward = 0.0;

  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    // std::cout << "i = " << It.GetIndex() << std::endl;
    // std::cout << " res_backward_2D = " << res_backward_2D->GetPixel(It.GetIndex()) << std::endl;
    // std::cout << " res_forward_2D = " << res_forward_2D->GetPixel(It.GetIndex()) << std::endl;
    // std::cout << " SingleDot_2D = " << It.Get() << std::endl;
    sum_backward  += res_backward_2D->GetPixel(It.GetIndex()) * It.Get();
    sum_forward   += res_forward_2D->GetPixel(It.GetIndex()) * It.Get();
    }

  std::cout << "| (Ax,y) - (x,A'y) | = " << std::abs(sum_forward-sum_backward) << std::endl;

  // ImageRegion<2> region;


  return EXIT_SUCCESS;
}
 