/*! \brief An adapted Resample Image Filter is coded based on the official ITK Examples.
 *
 *  The aim is to implement the slice-acquisition model to simulate one single
 *  given a current HR volume. The blurring operator (representing the PSF)
 *  shall be modelled as 3D Gaussian.
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date January 2016
 */

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"
#include <itkNiftiImageIO.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkGaussianInterpolateImageFunction.h>

#include <string>
#include <limits.h> /* PATH_MAX */
#include <math.h>

/* Does not work; Needs some debubbing */
// template <typename TParametersValueType, typename NDimensions>
// itk::AffineTransform<TParametersValueType, NDimensions>::Pointer 
// getAffineTransform(const TParametersValueType angleInDegrees, const InputImageType * outputImage){

//   /* Affine Transform */
//   typedef itk::AffineTransform< TParametersValueType, Dimension >  TransformType;
//   TransformType::Pointer transform = TransformType::New();


//   const InputImageType::SpacingType & spacing = outputImage->GetSpacing();
//   const InputImageType::PointType & origin = outputImage->GetOrigin();
//   const InputImageType::SizeType size = outputImage->GetLargestPossibleRegion().GetSize();


//   const TParametersValueType degreesToRadians = std::atan(1.0) / 45.0;
//   const TParametersValueType angle = angleInDegrees * degreesToRadians;
  
//   const TParametersValueType imageCenterX = origin[0] + spacing[0] * size[0] / 2.0;
//   const TParametersValueType imageCenterY = origin[1] + spacing[1] * size[1] / 2.0;

//   TransformType::OutputVectorType translation1;
//   translation1[0] =   -imageCenterX;
//   translation1[1] =   -imageCenterY;

//   transform->Translate( translation1 );
//   transform->Rotate2D( -angle, false );

//   std::cout << "imageCenterX = " << imageCenterX << std::endl;
//   std::cout << "imageCenterY = " << imageCenterY << std::endl;
  

//   TransformType::OutputVectorType translation2;
//   translation2[0] =   imageCenterX;
//   translation2[1] =   imageCenterY;

//   transform->Translate( translation2, false );

//   return transform;
// }

template <typename ImageType> itk::Matrix<double,3,3> getSigmaPSF(ImageType *HR_volume, ImageType *slice);
template <typename ImageType> itk::Matrix<double,3,3> computeRotationMatrix(ImageType *HR_volume, ImageType *slice);


template <typename ImageType> itk::Matrix<double,3,3> getSigmaPSF(ImageType *HR_volume, ImageType *slice){

  const typename ImageType::SpacingType spacing = slice->GetSpacing();
  const typename ImageType::DirectionType U = computeRotationMatrix(HR_volume, slice);

  itk::Matrix<double,3,3> SigmaPSF;
  itk::Matrix<double,3,3> S;

  /* Blurring modelled as axis-aligned Gaussian */
  SigmaPSF(0,0) = pow(1.2*spacing[0], 2) / (8*log(2));
  SigmaPSF(0,1) = 0.;
  SigmaPSF(0,2) = 0.;

  SigmaPSF(1,0) = 0.;
  SigmaPSF(1,1) = pow(1.2*spacing[1], 2) / (8*log(2));
  SigmaPSF(1,2) = 0.;

  SigmaPSF(2,0) = 0.;
  SigmaPSF(2,1) = 0.;
  SigmaPSF(2,2) = pow(spacing[2], 2) / (8*log(2));

  /* Scaling matrix */
  S(0,0) = spacing[0];
  S(0,1) = 0.;
  S(0,2) = 0.;

  S(1,0) = 0.;
  S(1,1) = spacing[1];
  S(1,2) = 0.;

  S(2,0) = 0.;
  S(2,1) = 0.;
  S(2,2) = spacing[2];

  std::cout << "SigmaPSF = \n" << SigmaPSF << std::endl;
  std::cout << "U = \n" << U << std::endl;

  /* Rotate Gaussian relativ to main axis of the HR volume*/
  SigmaPSF = U * SigmaPSF * U.GetTranspose();
  std::cout << "SigmaPSF_aligned = \n" << SigmaPSF << std::endl;

  /* Scale rotated Gaussian */
  SigmaPSF = S * SigmaPSF.GetInverse() * S;
  std::cout << "SigmaPSF_aligned_scaled = \n" << SigmaPSF << std::endl;


  return SigmaPSF;
}


template <typename ImageType> itk::Matrix<double,3,3> computeRotationMatrix(ImageType *HR_volume, ImageType *slice){

  const typename ImageType::DirectionType direction_HR_volume = HR_volume->GetDirection();
  const typename ImageType::DirectionType direction_slice = slice->GetDirection();

  itk::Matrix<double,3,3> U;

  U = direction_HR_volume.GetTranspose();
  U = U * direction_slice;

  return U;
}

int main( int argc, char * argv[] )
{
  // if( argc < 4 )
  //   {
  //   std::cerr << "Usage: " << std::endl;
  //   std::cerr << argv[0] << "  inputImageFile  outputImageFile  degrees" << std::endl;
  //   return EXIT_FAILURE;
  //   }
 

  const std::string dir_input = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/data/";
  const std::string dir_output = "/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/cpp/ITK_Examples/MyFunctions/results/";

  // const std::string filename_HR_volume = "BrainWeb_2D.png";
  const std::string filename_HR_volume = "FetalBrain_reconstruction_4stacks.nii.gz";
  const std::string filename_stack = "FetalBrain_stack2_registered.nii.gz";
  const std::string filename_slice = "FetalBrain_stack2_registered_midslice.nii.gz";
  const std::string filename_output = "test_output.nii.gz";

  /* Define dimension of image and pixel types */
  const     unsigned int   Dimension = 3;
  typedef   double  InputPixelType;
  typedef   double  OutputPixelType;
  // typedef   unsigned char  InputPixelType;
  // typedef   unsigned char  OutputPixelType;

  /* Define types of input and outpyt image */
  typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
  typedef itk::Image< OutputPixelType, Dimension >   OutputImageType;

  /* Define types of reader and writer */
  typedef itk::ImageFileReader< InputImageType  >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  
  /* Instantiate reader and writer */
  ReaderType::Pointer reader_HR_volume = ReaderType::New();
  ReaderType::Pointer reader_stack = ReaderType::New();
  ReaderType::Pointer reader_slice = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  /* Set image IO type to nifti */
  itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();  
  reader_HR_volume->SetImageIO(imageIO);
  reader_stack->SetImageIO(imageIO);
  reader_slice->SetImageIO(imageIO);

  /* Read images */
  reader_HR_volume->SetFileName( dir_input + filename_HR_volume );
  reader_HR_volume->Update();

  reader_stack->SetFileName( dir_input + filename_stack );
  reader_stack->Update();

  reader_slice->SetFileName( dir_input + filename_slice );
  reader_slice->Update();

  /* Get images */
  const InputImageType* HR_volume = reader_HR_volume->GetOutput();
  const InputImageType* stack = reader_stack->GetOutput();
  const InputImageType* slice = reader_slice->GetOutput();

  /* Resample Image Filter */
  typedef itk::ResampleImageFilter< InputImageType, OutputImageType >  FilterType;
  FilterType::Pointer filter = FilterType::New();

  /* Set input image */
  filter->SetInput(HR_volume);

  /* Set parameters of output image */
  const InputImageType * outputImage = slice;

  const InputImageType::SpacingType spacing = outputImage->GetSpacing();
  const InputImageType::PointType origin = outputImage->GetOrigin();
  const InputImageType::DirectionType direction = outputImage->GetDirection();
  const InputImageType::SizeType size = outputImage->GetLargestPossibleRegion().GetSize();

  filter->SetOutputOrigin( origin );
  filter->SetOutputSpacing( spacing );
  filter->SetOutputDirection( direction );
  filter->SetSize( size );

  /* Choose interpolator */
  // Linear Interpolator
  // typedef itk::LinearInterpolateImageFunction< InputImageType, double >  InterpolatorType;
  // InterpolatorType::Pointer interpolator = InterpolatorType::New();

  // Nearest Neighbour
  // typedef itk::NearestNeighborInterpolateImageFunction< InputImageType, double >  InterpolatorType;
  // InterpolatorType::Pointer interpolator = InterpolatorType::New();

  // Gaussian Interpolation
  const double alpha = 3;
  const itk::Matrix<double,3,3> SigmaPSF = getSigmaPSF(HR_volume, slice);

  // for (int i = 0; i < Dimension; ++i)
  // {
  //   std::cout << "SigmaPSF[" << i << "] = " << SigmaPSF[i] << std::endl;
  // }

  typedef itk::GaussianInterpolateImageFunction< InputImageType, double >  InterpolatorType;  
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetAlpha(alpha);
  // interpolator->SetSigma(SigmaPSF);

  /* Set interpolator */
  filter->SetInterpolator( interpolator );
  filter->SetDefaultPixelValue( 0 );  

  /* Affine Transform */
  typedef itk::AffineTransform< double, Dimension >  TransformType;
  TransformType::Pointer transform = TransformType::New();
  // TransformType::Pointer transform = getAffineTransform<double, Dimension>(angleInDegrees, inputImage);
  // filter->SetTransform( transform );


  try{
    /* Write images */
    writer->SetFileName( dir_output + filename_output );
    writer->SetInput( filter->GetOutput() );
    writer->Update();
  }
  catch( itk::ExceptionObject & excep ){
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
  }

  return EXIT_SUCCESS;
}
