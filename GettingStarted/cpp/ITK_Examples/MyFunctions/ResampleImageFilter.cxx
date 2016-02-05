/*! \brief An adapted Resample Image Filter is coded based on the official ITK Examples.
 *
 *  The aim is to implement the slice-acquisition model to simulate one single slice
 *  given a current HR volume. The slice/stack is already registered to the volume. 
 *  Hence, the blurring and downsampling step shall be performed in here whereby
 *  the blurring operator (representing the PSF) is modelled as 3D Gaussian.
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
#include <itkOrientedGaussianInterpolateImageFunction.h>

#include <string>
#include <limits.h> /* PATH_MAX */
#include <math.h>

/// TODO: Put lines below into file header!!

//! Compute oriented Gaussian kernel given the relative position of slice and HR volume
/*! 
  Given the relativ position of the slice to the HR volume a 3D Gaussian kernel
  is oriented such that the (rotated) PSF is expressed in the coordinate
  system of the HR volume (where the interpolation happens when specified as
  floating/source/moving image)

  \param[in] HR_volume PSF is oriented relativ to HR volume coordinate system
  \param[in] slice PSF is defined by slice dimensions and postition
  \return 3x3 matrix defining the rotated covariance matrix representing the PSF
 */
template <typename ImageType> itk::Matrix<double,3,3> 
  getOrientedSigmaPSF(ImageType *HR_volume, ImageType *slice);

//! Compute rotation matrix such that slice-defined PSF is expressed in coordinate system of HR volume
/*! 
  \param[in] HR_volume PSF is oriented relativ to HR volume coordinate system
  \param[in] slice PSF is defined by slice dimensions and postition
 */
template <typename ImageType> itk::Matrix<double,3,3> 
  computeRotationMatrix(ImageType *HR_volume, ImageType *slice);

//! Compute argument of exponential defining 3D Gaussian
/*!
  Evaluated function reads

    exp( -0.5* (point-center)'*Sigma(point-center) )

  Note: Sigma and not Sigma^{-1}!

  \param[in] point 3D gaussian pdf proportional to this point
  \param[in] center mean of 3D multivariate Gaussian
  \param[in] Sigma Inverse covariance matrix of 3D multivariate Gaussian
  \return evaluated exponential (scalar value)
 */
double 
  computeExponential(itk::Vector<double, 3> point, 
    itk::Vector<double, 3> center, itk::Matrix<double, 3, 3> Sigma); 


//! Returns 3D rotation transform given by angle
/*!
  TODO: Function just preliminary; Not generalized for arbitrary angles in
  each respective dimension

  \param[in] angleInDegrees Specifying rotation
  \param[in] Image Image exhibiting same dimensions/spacing/orientation as
             image to be rotated
  \return get orthogonal 3x3 matrix defining the rotation
*/
template <typename ImageType> itk::AffineTransform<double, 3>::Pointer 
  getRotationTransform3D(const double angleInDegrees, const ImageType *Image);






/// TODO: Put into cxx file

//! Compute oriented Gaussian kernel given the relative position of slice and HR volume
/*! 
  Given the relativ position of the slice to the HR volume a 3D Gaussian kernel
  is oriented such that the (rotated) PSF is expressed in the coordinate
  system of the HR volume (where the interpolation happens when specified as
  floating/source/moving image)

  \param[in] HR_volume PSF is oriented relativ to HR volume coordinate system
  \param[in] slice PSF is defined by slice dimensions and postition
  \return 3x3 matrix defining the rotated covariance matrix representing the PSF
 */
template <typename ImageType> itk::Matrix<double,3,3> 
  getOrientedSigmaPSF(ImageType *HR_volume, ImageType *slice){

  const typename ImageType::SpacingType spacing = slice->GetSpacing();
  const typename ImageType::DirectionType U = computeRotationMatrix(HR_volume, slice);

  itk::Matrix<double,3,3> SigmaPSF;
  SigmaPSF.Fill(0.0);

  /* Blurring modelled as axis-oriented Gaussian */
  SigmaPSF(0,0) = pow(1.2*spacing[0], 2) / (8*log(2));
  SigmaPSF(1,1) = pow(1.2*spacing[1], 2) / (8*log(2));
  SigmaPSF(2,2) = pow(spacing[2], 2) / (8*log(2));

  std::cout << "SigmaPSF = \n" << SigmaPSF << std::endl;
  std::cout << "U = \n" << U << std::endl;

  /* Rotate Gaussian relativ to main axis of the HR volume*/
  SigmaPSF = U * SigmaPSF * U.GetTranspose();
  std::cout << "SigmaPSF_oriented = \n" << SigmaPSF << std::endl;

  return SigmaPSF;
}

//! Compute rotation matrix such that slice-defined PSF is expressed in coordinate system of HR volume
/*! 
  \param[in] HR_volume PSF is oriented relativ to HR volume coordinate system
  \param[in] slice PSF is defined by slice dimensions and postition
  \return get orthogonal 3x3 matrix defining the rotation
 */
template <typename ImageType> itk::Matrix<double,3,3> 
  computeRotationMatrix(ImageType *HR_volume, ImageType *slice){

  const typename ImageType::DirectionType direction_HR_volume = HR_volume->GetDirection();
  const typename ImageType::DirectionType direction_slice = slice->GetDirection();

  itk::Matrix<double,3,3> U;

  U = direction_HR_volume.GetTranspose();
  U = U * direction_slice;

  return U;
}

//! Compute argument of exponential defining 3D Gaussian
/*!
  Evaluated function reads

    exp( -0.5* (point-center)'*Sigma(point-center) )

  Note: Sigma and not Sigma^{-1}!

  \param[in] point 3D gaussian pdf proportional to this point
  \param[in] center mean of 3D multivariate Gaussian
  \param[in] Sigma Inverse covariance matrix of 3D multivariate Gaussian
  \return evaluated exponential (scalar value)
 */
double 
  computeExponential(itk::Vector<double, 3> point, itk::Vector<double, 3> center, itk::Matrix<double, 3, 3> Sigma){
  
  itk::Vector<double, 3> tmp;
  double result;

  tmp = Sigma*(point-center);
  result = (point-center)*tmp;

  std::cout << "Sigma*(point-center) = " << tmp << std::endl;
  std::cout << "(point-center)'*Sigma*(point-center) = " << result << std::endl;

  result = exp(-0.5*result);
  std::cout << "exp(.) = " << result << std::endl;

  return result;

}

//! Returns 3D rotation transform given by angle
/*!
  TODO: Function just preliminary; Not generalized for arbitrary angles in
  each respective dimension

  \param[in] angleInDegrees Specifying rotation
  \param[in] Image Image exhibiting same dimensions/spacing/orientation as
             image to be rotated
  \return get orthogonal 3x3 matrix defining the rotation
*/
template <typename ImageType>
itk::AffineTransform<double, 3>::Pointer 
getRotationTransform3D(const double angleInDegrees, const ImageType *Image){

  // const int Dimension = Image->GetLargestPossibleRegion().GetImageDimension(); // does not work then
  const int Dimension = 3;

  /* Instantiate transform */
  typedef itk::AffineTransform< double, Dimension >  TransformType;
  typename TransformType::Pointer transform = TransformType::New();

  /* Get image parameters */
  const typename ImageType::SpacingType spacing = Image->GetSpacing();
  const typename ImageType::PointType origin = Image->GetOrigin();
  const typename ImageType::SizeType size = Image->GetLargestPossibleRegion().GetSize();

  /* Compute offset for rotation */
  double *imageCenter = new double[Dimension];  
  typename TransformType::OutputVectorType translation1;
  typename TransformType::OutputVectorType translation2;
  
  for (int i = 0; i < Dimension; ++i){
    imageCenter[i] = origin[i] + spacing[i] * size[i] / 2.0;
    translation1[i] = -imageCenter[i]; 
    translation2[i] = imageCenter[i]; 

    // std::cout << "imageCenter[" << i << "] = " << imageCenter[i] << std::endl;
  }

  /* Rotation angle in rad */
  const double degreesToRadians = std::atan(1.0) / 45.0;
  const double angle = angleInDegrees * degreesToRadians;

  /* Define rotation around respective axis */
  typename TransformType::OutputVectorType axis;
  axis[0] = 1;
  axis[1] = 0;
  axis[2] = 0;

  /* Set transformations */
  transform->Translate( translation1 );
  transform->Rotate3D( axis, -angle, false );
  transform->Translate( translation2, false );

  return transform;
}



int main( int argc, char * argv[] )
{
  // if( argc < 4 )
  //   {
  //   std::cerr << "Usage: " << std::endl;
  //   std::cerr << argv[0] << "  inputImageFile  outputImageFile  degrees" << std::endl;
  //   return EXIT_FAILURE;
  //   } 

  /* Define input and output */
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

  /* Get oriented PSF */
  const itk::Matrix<double,3,3> SigmaPSF_oriented = getOrientedSigmaPSF(HR_volume, slice);

  /* Scaling matrix */
  const InputImageType::SpacingType spacing_HR_volume = HR_volume->GetSpacing();
  itk::Matrix<double,3,3> S;
  S.Fill(0.0);
  S(0,0) = spacing_HR_volume[0];
  S(1,1) = spacing_HR_volume[1];
  S(2,2) = spacing_HR_volume[2];
  std::cout << "S = \n" << S << std::endl;
  
  /* Scale rotated inverse Gaussian needed for exponential function */
  itk::Matrix<double, 3, 3> SigmaPSF_oriented_scaled = S * SigmaPSF_oriented.GetInverse() * S;
  std::cout << "SigmaPSF_oriented_scaled = \n" << SigmaPSF_oriented_scaled << std::endl;


  /* Compute Gaussian */
  itk::Vector<double, 3> point;
  itk::Vector<double, 3> center;

  point.Fill( 5.0 );
  center.Fill( 3.0 );
  double weight = computeExponential(point, center, SigmaPSF_oriented_scaled);
  std::cout << weight << std::endl;



  // Gaussian Interpolation
  const double alpha = 3;
  double Sigma[3];

  for (int i = 0; i < 3; ++i)
  {
    Sigma[i] = sqrt(SigmaPSF_oriented(i,i));
  }

  // typedef itk::GaussianInterpolateImageFunction< InputImageType, double >  InterpolatorType;  
  typedef itk::OrientedGaussianInterpolateImageFunction< InputImageType, double >  InterpolatorType;

  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetAlpha(alpha);
  
  interpolator->SetSigma(Sigma);
  // interpolator->SetCovariance(SigmaPSF_oriented);

  /* Set interpolator */
  filter->SetInterpolator( interpolator );
  filter->SetDefaultPixelValue( 0 );  





  /* Affine Transform */
  typedef itk::AffineTransform< double, Dimension >  TransformType;
  const double angleInDegrees = 90;
  TransformType::Pointer transform = getRotationTransform3D(angleInDegrees, HR_volume);
  // filter->SetTransform( transform );

  // std::cout << HR_volume->GetLargestPossibleRegion().GetImageDimension() << std::endl;
  // std::cout << HR_volume->GetLargestPossibleRegion().GetSize().GetSizeDimension() << std::endl;

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
