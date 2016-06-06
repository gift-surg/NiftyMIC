/*! \brief Code to play with itk::N4BiasFieldCorrectionImageFilter
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date May 2016
 */

#include <string>
#include <limits.h>     /* PATH_MAX */
#include <math.h>
#include <cstdlib>     /* system, NULL, EXIT_FAILURE */
#include <chrono>

#include <itkImage.h>
#include "itkN4BiasFieldCorrectionImageFilter.h"
#include "itkOtsuThresholdImageFilter.h"

// My includes
#include "MyITKImageHelper.h"
#include "readCommandLine.h"
#include "MyException.h"

// Typedefs
typedef itk::OtsuThresholdImageFilter<ImageType3D, MaskImageType3D> ThresholderType3D;
typedef itk::N4BiasFieldCorrectionImageFilter<ImageType3D, MaskImageType3D, ImageType3D> CorrecterType3D;

int main(int argc, char** argv)
{

    //***Parse input of command line
    const std::vector<std::string> input = readCommandLine(argc, argv);

    //***Check for empty vector ==> It was given "--help" in command line
    if( input[0] == "help request" ){
      return EXIT_SUCCESS;
    }

    //***Read input data of command line
    const std::string sImage = input[0];        //"fixed" --f
    const std::string sImageMask = input[2];    //"fixed mask" --fmask

    const std::string sDirOut = input[17];      //"transformout" --tout
    const std::string sFilenameOut = input[1];  //"moving" --m
    const std::string sImageCorrected = sFilenameOut + "_corrected";

    const bool bVerbose = std::stoi(input[19]);


    MaskImageType3D::Pointer image_3D_mask;
    // Read images
    const ImageType3D::Pointer image_3D = MyITKImageHelper::readImage<ImageType3D>(sImage + ".nii.gz");
    if ( bVerbose ){
        std::cout << "image  = " << sImage << std::endl;
    }

    // Read masks
    if(!sImageMask.empty()){
        if ( bVerbose ){
            std::cout << "image mask = " << sImageMask << std::endl;
        } 
      image_3D_mask = MyITKImageHelper::readImage<MaskImageType3D>(sImageMask + ".nii.gz");
    }
    if ( bVerbose ){
        std::cout << "write corrected image to " << sDirOut + sImageCorrected << std::endl;
    }

    // MyITKImageHelper::showImage(image_3D, image_3D_mask);

    /*
        N4 Bias Field Correction Image Filter
    */
    // MaskImageType3D::Pointer image_3D_mask = ITK_NULLPTR;

    if( !image_3D_mask ){
        std::cout << "Mask not read.  Creating Otsu mask." << std::endl;
        ThresholderType3D::Pointer otsu = ThresholderType3D::New();

        otsu->SetInput( image_3D );
        // otsu->SetNumberOfHistogramBins( 200 );
        otsu->SetInsideValue( 0 );
        otsu->SetOutsideValue( 1 );

        otsu->Update();
        image_3D_mask = otsu->GetOutput();
        image_3D_mask->DisconnectPipeline();
    }
    // MyITKImageHelper::showImage(image_3D, image_3D_mask, "inputImage_mask");


    const unsigned int iFittingLevels = 4;
    CorrecterType3D::VariableSizeArrayType  maximumNumberOfIterations( iFittingLevels );
    maximumNumberOfIterations.Fill( 50 );

    CorrecterType3D::Pointer correcter = CorrecterType3D::New();
    correcter->SetMaximumNumberOfIterations( maximumNumberOfIterations );
    correcter->SetNumberOfFittingLevels( iFittingLevels );
    correcter->SetConvergenceThreshold( 1e-6 );

    correcter->SetMaskLabel( 1 );
    correcter->SetSplineOrder( 3 );
    correcter->SetWienerFilterNoise( 0.11 );
    correcter->SetBiasFieldFullWidthAtHalfMaximum( 0.15 );

    // set the input image and mask image
    correcter->SetInput( image_3D );
    correcter->SetMaskImage( image_3D_mask );

    try{
        correcter->Update();
    }
    catch( itk::ExceptionObject &excep ){
        std::cerr << "Exception caught !" << std::endl;
        std::cerr << excep << std::endl;
        return EXIT_FAILURE;
    }

    const ImageType3D::Pointer image_3D_corrected = correcter->GetOutput();
    image_3D_corrected->DisconnectPipeline();

    // MyITKImageHelper::showImage(image_3D, image_3D_corrected, "imageOrig_imageCorr");
    // MyITKImageHelper::writeImage(image_3D, sDirOut + sFilenameOut + ".nii.gz");
    MyITKImageHelper::writeImage(image_3D_corrected, sDirOut + sFilenameOut + "_corrected.nii.gz");

    return EXIT_SUCCESS;
}
 