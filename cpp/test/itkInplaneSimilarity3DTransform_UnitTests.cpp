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

typedef itk::AbsoluteValueDifferenceImageFilter< ImageType2D, ImageType2D, ImageType2D> AbsoluteValueDifferenceImageFilter_2D;
typedef itk::AbsoluteValueDifferenceImageFilter< ImageType3D, ImageType3D, ImageType3D> AbsoluteValueDifferenceImageFilter_3D;

typedef itk::StatisticsImageFilter<ImageType2D> StatisticsImageFilterType_2D;
typedef itk::StatisticsImageFilter<ImageType3D> StatisticsImageFilterType_3D;

typedef itk::Euler3DTransform< PixelType > EulerTransformType;
typedef itk::InplaneSimilarity3DTransform< PixelType > InplaneSimilarityTransformType;

// Unit tests
TEST_CASE( "itkInplaneSimilarity3DTransform: Brain", 
  "[itkInplaneSimilarity3DTransform: Brain]") {

    // Define input and output
    const std::string dir_input = "../exampleData/";

    const std::string filename = "FetalBrain_reconstruction_3stacks_myAlg.nii.gz";

    const double tolerance = 1e-6;

    // Read images
    const ImageType3D::Pointer image = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename);
    const ImageType3D::Pointer image_registered = MyITKImageHelper::readImage<ImageType3D>(dir_input + filename);

    MyITKImageHelper::showImage(image, "image");

    // Resample Image Filter
    const FilterType_Resample_3D::Pointer resampler = FilterType_Resample_3D::New();

    resampler->SetInput(image_registered);
    resampler->SetOutputParametersFromImage(image);
    resampler->Update();

    // Filters to evaluate absolute difference
    const MultiplyImageFilter_3D::Pointer multiplyFilter_3D = MultiplyImageFilter_3D::New();
    const AddImageFilter_3D::Pointer addFilter_3D = AddImageFilter_3D::New();
    const AbsoluteValueDifferenceImageFilter_3D::Pointer absDiffFilter_3D = AbsoluteValueDifferenceImageFilter_3D::New();
    const StatisticsImageFilterType_3D::Pointer statisticsImageFilter_3D = StatisticsImageFilterType_3D::New();

    multiplyFilter_3D->SetInput( image_registered );
    multiplyFilter_3D->SetConstant( -1 );

    addFilter_3D->SetInput1( image );
    addFilter_3D->SetInput2( multiplyFilter_3D->GetOutput() );
    
    statisticsImageFilter_3D->SetInput( addFilter_3D->GetOutput() );
    statisticsImageFilter_3D->Update();
    const double diff = statisticsImageFilter_3D->GetSum();

    std::cout << "Difference = " << diff << std::endl;

    CHECK( diff == Approx(0).epsilon(tolerance));
}
