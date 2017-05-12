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


TEST_CASE( "Check GradientEuler3DTransformImageFilter", "[GradientEuler3DTransformImageFilter]"){
    
    // Define input and output
    const std::string dir_input = "../test-data/";

    const std::string filename_slice = "FetalBrain_stack2_registered_midslice.nii.gz";

    const double tolerance = 1e-6;
    const int DOFS = 6;
    const int DIMENSION = 3;

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
    itk::Point<PixelType, DIMENSION> point;
    ImageType3D::RegionType region = slice->GetBufferedRegion();
    itk::ImageRegionConstIteratorWithIndex<ImageType3D> it( slice, region );

    // Variables for comparison
    typedef itk::CovariantVector< PixelType, DIMENSION*DOFS> CovariantVectorType;
    typedef itk::Image< CovariantVectorType, DIMENSION > ImageCovariantVectorType;

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
        for (int i = 0; i < DIMENSION; ++i) {
            for (int j = 0; j < DOFS; ++j) {

                double abs_diff = std::abs(jacobian[i*DOFS + j] - jacobian2.GetElement(i,j));

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
        for (int i = 0; i < DIMENSION; ++i) {
            for (int j = 0; j < DOFS; ++j) {

                double abs_diff = std::abs(jacobian[i*DOFS + j] - jacobian2.GetElement(i,j));

                CHECK( abs_diff == Approx(0).epsilon(tolerance));
            }
        }

        ++it;
    }


}