/*! \brief
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date May 2016
 */
#include <boost/type_traits.hpp>

#include <iostream>
#include <string>
#include <limits.h>     /* PATH_MAX */
#include <math.h>
#include <cstdlib>     /* system, NULL, EXIT_FAILURE */
#include <chrono>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <itkImageRegistrationMethodv4.h>
#include <itkCenteredTransformInitializer.h>

#include <itkInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkBSplineInterpolateImageFunction.h>

#include <itkImageToImageMetricv4.h>
#include <itkMeanSquaresImageToImageMetricv4.h>
#include <itkMattesMutualInformationImageToImageMetricv4.h>
#include <itkCorrelationImageToImageMetricv4.h>
#include <itkANTSNeighborhoodCorrelationImageToImageMetricv4.h>

#include <itkRegularStepGradientDescentOptimizerv4.h>
#include <itkLBFGSBOptimizerv4.h>

#include <itkResampleImageFilter.h>
// #include <itkRescaleIntensityImageFilter.h>

#include <itkAffineTransform.h>
#include <itkEuler3DTransform.h>
#include <itkImageMaskSpatialObject.h>

#include <itkRegistrationParameterScalesFromJacobian.h>
#include <itkRegistrationParameterScalesFromIndexShift.h>
#include <itkRegistrationParameterScalesFromPhysicalShift.h>

#include <itkCommand.h>

// My includes
#include "MyITKImageHelper.h"
#include "itkOrientedGaussianInterpolateImageFunction.h"
#include "readCommandLine.h"
#include "MyException.h"
#include "itkScaledTranslationEuler3DTransform.h"

// Global variables
const unsigned int Dimension = 3;

// Typedefs 
typedef itk::ResampleImageFilter< ImageType3D, ImageType3D > ResampleFilterType;
typedef itk::ResampleImageFilter< MaskImageType3D, MaskImageType3D > MaskResampleFilterType;

typedef itk::ImageMaskSpatialObject< Dimension > MaskType;

// Transform Types
typedef itk::AffineTransform< PixelType, Dimension > AffineTransformType;
typedef itk::ScaledTranslationEuler3DTransform< PixelType > EulerTransformType;
// typedef itk::Euler3DTransform< PixelType > EulerTransformType;

// Optimizer Types
typedef itk::RegularStepGradientDescentOptimizerv4< PixelType > RegularStepGradientDescentOptimizerType;
typedef itk::LBFGSBOptimizerv4 LBFGSBOptimizerOptimizerType;
// typedef RegularStepGradientDescentOptimizerType OptimizerType;
typedef LBFGSBOptimizerOptimizerType OptimizerType;

// Interpolator Types
typedef itk::NearestNeighborInterpolateImageFunction< ImageType3D, PixelType > NearestNeighborInterpolatorType;
typedef itk::LinearInterpolateImageFunction< ImageType3D, PixelType > LinearInterpolatorType;
typedef itk::BSplineInterpolateImageFunction< ImageType3D, PixelType > BSplineInterpolatorType;
typedef itk::OrientedGaussianInterpolateImageFunction< ImageType3D, PixelType >  OrientedGaussianInterpolatorType;

// Metric Types
typedef itk::MeanSquaresImageToImageMetricv4< ImageType3D, ImageType3D > MeanSquaresMetricType;
typedef itk::CorrelationImageToImageMetricv4< ImageType3D, ImageType3D > CorrelationMetricType;
typedef itk::MattesMutualInformationImageToImageMetricv4< ImageType3D, ImageType3D > MattesMutualInformationMetricType;
typedef itk::ANTSNeighborhoodCorrelationImageToImageMetricv4 <ImageType3D, ImageType3D> ANTSNeighborhoodCorrelationMetricType;
// Scales Estimator Types (based on Metric and need to be set depending on them)



// itk::InterpolateImageFunction<ImageType3D, PixelType>* getInterpolator(std::string &sInterpolator, const itk::Vector<double, 9> covariance, const double alpha){
    
//     if ( sInterpolator == ("NearestNeighbor") ) {
//         std::cout << "Chosen interpolator is " << sInterpolator << std::endl;
//         return NearestNeighborInterpolatorType::New();
//     }

//     else if ( sInterpolator == ("Linear") ) {
//         std::cout << "Chosen interpolator is " << sInterpolator << std::endl;
//         LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();

//         return interpolator.GetPointer();
//     }

//     else if ( sInterpolator == ("BSpline") ) {
//         std::cout << "Chosen interpolator is " << sInterpolator << std::endl;
//         BSplineInterpolatorType::Pointer interpolator = BSplineInterpolatorType::New();

//         return interpolator.GetPointer();
//     }

//     else if ( sInterpolator == ("OrientedGaussian") ) {
//         std::cout << "Chosen interpolator is " << sInterpolator << std::endl;
//         OrientedGaussianInterpolatorType::Pointer interpolator = OrientedGaussianInterpolatorType::New();
        
//         interpolator->SetCovariance( covariance );
//         interpolator->SetAlpha( alpha );
        
//         return interpolator.GetPointer();
//     }

//     else {
//         std::cout << sInterpolator << " cannot be deduced." << std::endl;
//         return NULL;
//     }
// }


// itk::ImageToImageMetricv4<ImageType3D, ImageType3D>::Pointer getMetric(const std::string &sMetric){
//     itk::ImageToImageMetricv4<ImageType3D, ImageType3D>::Pointer metric = 0;

//     if ( sMetric == ("MeanSquares") ) {
//         std::cout << "Chosen metric is " << sMetric << std::endl;
//         metric = MeanSquaresMetricType::New();
//     }

//     else if ( sMetric == ("Correlation") ) {
//         std::cout << "Chosen metric is " << sMetric << std::endl;
//         metric = CorrelationMetricType::New();
//     }

//     else if ( sMetric == ("MattesMutualInformation") ) {
//         std::cout << "Chosen metric is " << sMetric << std::endl;
//         metric = MattesMutualInformationMetricType::New();
//     }

//     else {
//         std::cout << sMetric << " cannot be deduced." << std::endl;
//         metric = NULL;
//     }

//     return metric;
// }

class CommandIterationUpdate : public itk::Command
{
    public:
        typedef  CommandIterationUpdate   Self;
        typedef  itk::Command             Superclass;
        typedef  itk::SmartPointer<Self>  Pointer;
        itkNewMacro( Self );
    protected:
        CommandIterationUpdate(): m_CumulativeIterationIndex(0) {};
    public:
        // typedef   itk::RegularStepGradientDescentOptimizerv4<double>  OptimizerType;
        typedef   const OptimizerType *                               OptimizerPointer;
        void Execute(itk::Object *caller, const itk::EventObject & event) ITK_OVERRIDE {
            Execute( (const itk::Object *)caller, event);
        }
        void Execute(const itk::Object * object, const itk::EventObject & event) ITK_OVERRIDE {
            OptimizerPointer optimizer = static_cast< OptimizerPointer >( object );
            if( !(itk::IterationEvent().CheckEvent( &event )) ) {
                return;
            }

            std::cout << "iteration cost [parameters] CumulativeIterationIndex" << std::endl;
            std::cout << optimizer->GetCurrentIteration() << "   ";
            std::cout << optimizer->GetValue() << "   ";
            std::cout << optimizer->GetCurrentPosition() << "   ";
            std::cout << m_CumulativeIterationIndex++ << std::endl;
        }
    private:
        unsigned int m_CumulativeIterationIndex;
};


template <typename TransformType, typename InterpolatorType, typename MetricType, typename ScalesEstimatorType  >
void RegistrationFunction( const std::vector<std::string> &input ) {

    // Image Registration Type    
    typedef itk::ImageRegistrationMethodv4< ImageType3D, ImageType3D, TransformType > RegistrationType;

    // Centered Transform Initializer
    typedef itk::CenteredTransformInitializer< TransformType, ImageType3D, ImageType3D > TransformInitializerType;

    const bool bAddObserver = false;
    const std::string sBar = "------------------------------------------------------" 
        "----------------------------\n";

    //*** Define option variables
    bool bUseMovingMask = false;
    bool bUseFixedMask = false;
    bool bUseMultiresolution = false;

    ///***Instantiate
    const typename RegistrationType::Pointer registration = RegistrationType::New();
    const typename MetricType::Pointer metric = MetricType::New();
    const typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    const OptimizerType::Pointer optimizer = OptimizerType::New();
    const typename ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    
    const MaskType::Pointer spatialObjectFixedMask = MaskType::New();
    const MaskType::Pointer spatialObjectMovingMask = MaskType::New();
    MaskImageType3D::Pointer fixedMask;
    MaskImageType3D::Pointer movingMask;

    const unsigned int numberOfLevels = 3;
    typename RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    typename RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;

    //***Read input data of command line
    const std::string sFixed = input[0];
    const std::string sMoving = input[1];
    const std::string sFixedMask = input[2];
    const std::string sMovingMask = input[3];

    // Oriented Gaussian Interpolator
    const unsigned int alpha = 3;
    itk::Vector<double, 9> covariance;
    for (int i = 0; i < 9; ++i) {
        covariance[i] = std::stod(input[4+i]);
    } 
    // covariance.Fill(0);
    // covariance[0] = 0.26786367;
    // covariance[4] = 0.26786367;
    // covariance[8] = 2.67304559;

    const std::string sUseMultiresolution = input[13];
    const std::string sUseAffine = input[14];
    const std::string sMetric = input[15];
    const std::string sInterpolator = input[16];
    const std::string sTransformOut = input[17];
    const std::string sVerbose = input[19]; //TODO: change to bVerbose directly
    const double dANTSrad = std::stod(input[20]);

    // Read images
    const ImageType3D::Pointer moving = MyITKImageHelper::readImage<ImageType3D>(sMoving + ".nii.gz");
    const ImageType3D::Pointer fixed = MyITKImageHelper::readImage<ImageType3D>(sFixed + ".nii.gz");
    std::cout << "Fixed image  = " << sFixed << std::endl;
    std::cout << "Moving image = " << sMoving << std::endl;

    // MyITKImageHelper::showImage(moving, "moving");
    // MyITKImageHelper::showImage(fixed, fixedMask, "fixed");

    // Read masks
    if(!sFixedMask.empty()){
        std::cout << "Fixed mask image = " << sFixedMask << std::endl;
        bUseFixedMask = true;
        fixedMask = MyITKImageHelper::readImage<MaskImageType3D>(sFixedMask + ".nii.gz");
        spatialObjectFixedMask->SetImage( fixedMask );
        metric->SetFixedImageMask( spatialObjectFixedMask );
    }
    if(!sMovingMask.empty()){
        std::cout << "Moving mask image = " << sMovingMask << std::endl;
        bUseMovingMask = true;
        movingMask = MyITKImageHelper::readImage<MaskImageType3D>(sMovingMask + ".nii.gz");
        spatialObjectMovingMask->SetImage( movingMask );
        metric->SetMovingImageMask( spatialObjectMovingMask );
    }
    
    // Multi-resolution framework
    if(std::stoi(sUseMultiresolution)) {
        bUseMultiresolution = true;
        std::cout << "Multiresolution framework used" << std::endl;
        
        shrinkFactorsPerLevel.SetSize( numberOfLevels );
        shrinkFactorsPerLevel[0] = 4;
        shrinkFactorsPerLevel[1] = 2;
        shrinkFactorsPerLevel[2] = 1;

        smoothingSigmasPerLevel.SetSize( numberOfLevels );
        smoothingSigmasPerLevel[0] = 2;
        smoothingSigmasPerLevel[1] = 1;
        smoothingSigmasPerLevel[2] = 0;

        registration->SetNumberOfLevels ( numberOfLevels );
        registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
        registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
    }   


    // typename MetricType::MeasureType valueReturn;
    // typename MetricType::DerivativeType derivativeReturn;
    ANTSNeighborhoodCorrelationMetricType::Pointer ANTSmetric = dynamic_cast< ANTSNeighborhoodCorrelationMetricType* >(metric.GetPointer());
    if ( ANTSmetric.IsNotNull() ) {
        // set all parameters
        itk::Size<Dimension> neighborhoodRadius; 
        neighborhoodRadius.Fill(dANTSrad); 
        ANTSmetric->SetRadius(neighborhoodRadius);
        ANTSmetric->SetFixedImage(fixed);
        ANTSmetric->SetMovingImage(moving);
        ANTSmetric->SetFixedTransform(TransformType::New());
        ANTSmetric->SetMovingTransform(TransformType::New());
        // initialization after parameters are set
        ANTSmetric->Initialize();
        std::cout << "Radius for ANTSNeighborhoodCorrelation = " << dANTSrad << std::endl;
        // getting derivative and metric value
        // ANTSmetric->GetValueAndDerivative(valueReturn, derivativeReturn);
    }

    // Set oriented Gaussian interpolator (if given)
    OrientedGaussianInterpolatorType::Pointer orientedGaussianInterpolator = dynamic_cast< OrientedGaussianInterpolatorType* >(interpolator.GetPointer());
    if ( orientedGaussianInterpolator.IsNotNull() ) {
    // if ( sInterpolator == ("OrientedGaussian") ) {
        orientedGaussianInterpolator->SetCovariance( covariance );
        orientedGaussianInterpolator->SetAlpha( 3 );
        // std::cout << "OrientedGaussianInterpolator updated " << std::endl;
        std::cout << "covariance for oriented Gaussian = " << std::endl;
        for (int i = 0; i < 3; ++i) {
            printf("\t%.3f\t%.3f\t%.3f\n", covariance[3*i], covariance[3*i+1], covariance[3*i+2]);
        }
    }

    // Sort of "Debug". Not convenient since all is printed. Only type name would be great to see and test for
    // std::cout << sBar;
    // TransformType::New()->Print(std::cout);
    // std::cout << sBar;
    // interpolator->Print(std::cout);
    // std::cout << sBar;
    // metric->Print(std::cout);
    // std::cout << sBar;
    // scalesEstimator->Print(std::cout);
    // std::cout << sBar;

    // Initialize the transform
    typename TransformType::Pointer initialTransform = TransformType::New();

    // typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();
    // typename TransformType::Pointer foo = TransformType::New();
    // initializer->SetTransform(initialTransform);
    // initializer->SetTransform(foo);
    // initializer->SetFixedImage( fixed );
    // initializer->SetMovingImage( moving );
    // initializer->GeometryOn();
    // initializer->MomentsOn();
    // initializer->InitializeTransform();
    // initialTransform->Print(std::cout);
    // initialTransform->SetTranslation((0,0,0));
    // initialTransform->Print(std::cout);
    // initialTransform->SetFixedParameters(foo->GetFixedParameters());    
    registration->SetFixedInitialTransform( initialTransform );
    registration->InPlaceOff();
    // registration->GetFixedInitialTransform()->Print(std::cout);

    // Set metric
    // metric->SetFixedInterpolator(  interpolator  );
    metric->SetMovingInterpolator(  interpolator  );

    // Scales estimator
    // scalesEstimator->SetTransformForward( true );
    // scalesEstimator->SetSmallParameterVariation( 1.0 );
    scalesEstimator->SetMetric( metric );

    // Parametrize optimizer
    // optimizer->SetMinimumStepLength( 1e-6 );
    // optimizer->SetGradientMagnitudeTolerance( 1e-4 );
    // optimizer->SetMaximumStepLength( 0.1 ); // If this is set too high, you will get a
    //"itk::ERROR: MeanSquaresImageToImageMetric(0xa27ce70): Too many samples map outside moving image buffer: 1818 / 10000" error
    // optimizer->SetNumberOfIterations( 100 );
    // optimizer->SetMinimumConvergenceValue( 1e-6 );
    // optimizer->SetScalesEstimator( scalesEstimator );
    // optimizer->SetDoEstimateLearningRateOnce( false );
    // optimizer->SetLearningRate(1);

    // For LBFGS Optimizer
    const unsigned int numParameters = initialTransform->GetNumberOfParameters();
    OptimizerType::BoundSelectionType boundSelect( numParameters );
    OptimizerType::BoundValueType upperBound( numParameters );
    OptimizerType::BoundValueType lowerBound( numParameters );
    boundSelect.Fill( OptimizerType::BOTHBOUNDED );
    upperBound.Fill( 0.0 );
    lowerBound.Fill( 0.0 );

    const double angle_deg_max = 5.0;
    const double translation_max = 20.0;
    for (int i = 0; i < 3; ++i) {
        lowerBound[i] = -angle_deg_max*vnl_math::pi/180;
        upperBound[i] =  angle_deg_max*vnl_math::pi/180;
        
        lowerBound[i+3] = -translation_max;
        upperBound[i+3] =  translation_max;
    }



    optimizer->SetBoundSelection( boundSelect );
    optimizer->SetUpperBound( upperBound );
    optimizer->SetLowerBound( lowerBound );

    optimizer->SetCostFunctionConvergenceFactor( 1.e7 );
    optimizer->SetGradientConvergenceTolerance( 1e-35 );
    optimizer->SetNumberOfIterations( 200 );
    optimizer->SetMaximumNumberOfFunctionEvaluations( 200 );
    optimizer->SetMaximumNumberOfCorrections( 7 );

    // optimizer->SetDefaultStepLength( 1.5 );
    // optimizer->SetGradientConvergenceTolerance( 5e-2 );
    // optimizer->SetLineSearchAccuracy( 1.2 );
    // optimizer->TraceOn();
    // optimizer->SetMaximumNumberOfFunctionEvaluations( 1000 );

    // Create the Command observer and register it with the optimizer.
    CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    if ( bAddObserver ) {
        optimizer->AddObserver( itk::IterationEvent(), observer );
    }

    // Set registration
    registration->SetFixedImage(fixed);
    registration->SetMovingImage(moving);
    
    registration->SetMetric( metric );
    registration->SetOptimizer( optimizer );

    //***Execute registration
    try {
      registration->Update();
      std::cout << "Optimizer stop condition: "
      << registration->GetOptimizer()->GetStopConditionDescription()
      << std::endl;
    }
    catch( itk::ExceptionObject & err ) {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      // return EXIT_FAILURE;
      throw MyException("ExeceptionObject caught during registration");
    }


    //***Process registration results
    typename TransformType::ConstPointer transform = registration->GetTransform();
    
    if ( std::stoi(sVerbose) ) {
        // transform->Print(std::cout);
        MyITKImageHelper::printTransform(transform);
    }


    //  The value of the image metric corresponding to the last set of parameters
    //  can be obtained with the \code{GetValue()} method of the optimizer.
    const double bestValue = optimizer->GetValue();
    
    // Print out results
    // std::cout << "Result:" << std::endl;
    // std::cout << "\tMetric value  = " << bestValue          << std::endl;

    //***Write result to file
    if ( !sTransformOut.empty() ) {
        MyITKImageHelper::writeTransform(transform, sTransformOut);
    }

    //***Resample warped moving image
    // Resampling
    const ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    const MaskResampleFilterType::Pointer resamplerMask = MaskResampleFilterType::New();
    
    // Resample registered moving image
    resampler->SetOutputParametersFromImage( fixed );
    // resampler->SetSize( fixed->GetLargestPossibleRegion().GetSize() );
    // resampler->SetOutputOrigin(  fixed->GetOrigin() );
    // resampler->SetOutputSpacing( fixed->GetSpacing() );
    // resampler->SetOutputDirection( fixed->GetDirection() );
    resampler->SetInput( moving );
    resampler->SetTransform( registration->GetOutput()->Get() );
    resampler->SetDefaultPixelValue( 0.0 );
    resampler->SetInterpolator( interpolator );
    resampler->Update();

    // Resample registered moving mask
    if ( bUseMovingMask && bUseFixedMask){
        resamplerMask->SetOutputParametersFromImage( fixedMask );
        resamplerMask->SetInput( movingMask );
        resamplerMask->SetTransform( registration->GetOutput()->Get() );
        resamplerMask->SetDefaultPixelValue( 0.0 );
        resamplerMask->Update();
    }

    const ImageType3D::Pointer movingWarped = resampler->GetOutput();
    movingWarped->DisconnectPipeline();

    const MaskImageType3D::Pointer movingMaskWarped = resamplerMask->GetOutput();
    movingMaskWarped->DisconnectPipeline();

    // MyITKImageHelper::showImage(fixed, movingWarped, "fixed_moving");
    // MyITKImageHelper::showImage(movingWarped, movingMaskWarped, "fixed_mask");

}


int main(int argc, char** argv)
{

    try{

        //***Parse input of command line
        const std::vector<std::string> input = readCommandLine(argc, argv);

        //***Check for empty vector ==> It was given "--help" in command line
        if( input[0] == "help request" ){
            return EXIT_SUCCESS;
        }

        //***Read relevant input data to choose leaf node from command line
        const std::string sUseAffine = input[14];
        const std::string sMetric = input[15];
        const std::string sInterpolator = input[16];
        const std::string sScalesEstimator = input[18];

        // What the hell is that!?
        // std::string sInterpolatorTest = "BSpline";
        // std::cout << sInterpolatorTest << std::endl;
        // std::cout << sInterpolatorTest.compare("BSpline") << std::endl; // does not work
        // std::cout << (sInterpolatorTest == ("BSpline")) << std::endl;   // works

        // TODO: At the moment only rigid model is available
        switch ( std::stoi(sUseAffine) ){
            
            // Rigid registration
            case 0:
                std::cout << "Chosen type of registration: Rigid" << std::endl;

                // Nearest Neighbor interpolator
                if ( sInterpolator == ("NearestNeighbor") ) {
                    std::cout << "Chosen type of interpolator: " << sInterpolator << std::endl;

                    // Mean Squares metric
                    if ( sMetric == ("MeanSquares") ) { 
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MeanSquaresMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromIndexShift< MeanSquaresMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromJacobian< MeanSquaresMetricType > >(input);
                        }

                    }

                    // Normalized Cross Correlation Metric
                    else if ( sMetric == ("Correlation") ){
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< CorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< CorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< CorrelationMetricType > >(input);
                        }

                    }

                    // ANTS Neighborhood Correlation Metric
                    else if ( sMetric == ("ANTSNeighborhoodCorrelation") ){
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< ANTSNeighborhoodCorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                    }

                    // Mattes Mutual Information Metric
                    else {
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MattesMutualInformationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromIndexShift< MattesMutualInformationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromJacobian< MattesMutualInformationMetricType > >(input);
                        }
                    }
                }

                // Linear interpolator
                else if ( sInterpolator == ("Linear") ) {
                    std::cout << "Chosen type of interpolator: " << sInterpolator << std::endl;

                    // Mean Squares metric
                    if ( sMetric == ("MeanSquares") ) { 
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MeanSquaresMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromIndexShift< MeanSquaresMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromJacobian< MeanSquaresMetricType > >(input);
                        }

                    }

                    // Normalized Cross Correlation Metric
                    else if ( sMetric == ("Correlation") ){
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< CorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< CorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< CorrelationMetricType > >(input);
                        }

                    }

                    // ANTS Neighborhood Correlation Metric
                    else if ( sMetric == ("ANTSNeighborhoodCorrelation") ){
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< ANTSNeighborhoodCorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                    }

                    // Mattes Mutual Information Metric
                    else {
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MattesMutualInformationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromIndexShift< MattesMutualInformationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromJacobian< MattesMutualInformationMetricType > >(input);
                        }
                    }
                }

                // Oriented Gaussian interpolator
                else if ( sInterpolator == ("OrientedGaussian") ) {
                    std::cout << "Chosen type of interpolator: " << sInterpolator << std::endl;

                    // Mean Squares metric
                    if ( sMetric == ("MeanSquares") ) { 
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MeanSquaresMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromIndexShift< MeanSquaresMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromJacobian< MeanSquaresMetricType > >(input);
                        }

                    }

                    // Normalized Cross Correlation Metric
                    else if ( sMetric == ("Correlation") ){
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< CorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< CorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< CorrelationMetricType > >(input);
                        }

                    }

                    // ANTS Neighborhood Correlation Metric
                    else if ( sMetric == ("ANTSNeighborhoodCorrelation") ){
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< ANTSNeighborhoodCorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                    }

                    // Mattes Mutual Information Metric
                    else {
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MattesMutualInformationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromIndexShift< MattesMutualInformationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromJacobian< MattesMutualInformationMetricType > >(input);
                        }
                    }
                }

                // BSpline interpolator
                else {
                    std::cout << "Chosen type of interpolator: BSpline" << std::endl;

                    // Mean Squares metric
                    if ( sMetric == ("MeanSquares") ) { 
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MeanSquaresMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromIndexShift< MeanSquaresMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromJacobian< MeanSquaresMetricType > >(input);
                        }

                    }

                    // Normalized Cross Correlation Metric
                    else if ( sMetric == ("Correlation") ){
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< CorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< CorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< CorrelationMetricType > >(input);
                        }

                    }

                    // ANTS Neighborhood Correlation Metric
                    else if ( sMetric == ("ANTSNeighborhoodCorrelation") ){
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< ANTSNeighborhoodCorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                    }

                    // Mattes Mutual Information Metric
                    else {
                        std::cout << "Chosen type of metric: " << sMetric << std::endl;

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;
                            
                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MattesMutualInformationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            std::cout << "Chosen type of scales estimator: " << sScalesEstimator << std::endl;

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromIndexShift< MattesMutualInformationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            std::cout << "Chosen type of scales estimator: Jacobian"  << std::endl;

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromJacobian< MattesMutualInformationMetricType > >(input);
                        }
                    }
                }

                break;


            // TODO: Same as above but replace EulerTransformType by AffineTransformType.
            // However, write test cases first!!!
            // case 1:
            //     std::cout << "Affine registration used" << std::endl;
            //     RegistrationFunction<AffineTransformType, MeanSquaresMetricType >(input);
            //     break;

            default:

                // // Scales Estimator Types
                // typedef  PhysicalShiftScalesEstimatorType;
                // typedef itk::RegistrationParameterScalesFromIndexShift< MetricType > IndexShiftScalesEstimatorType;
                // typedef itk::RegistrationParameterScalesFromJacobian< MetricType > JacobianScalesEstimatorType;

                std::cout << "Chosen type of registration: Rigid" << std::endl;
                std::cout << "Chosen type of interpolator: BSpline" << std::endl;
                std::cout << "Chosen type of metric: Mattes Mutual Information" << std::endl;
                std::cout << "Chosen type of scales Estimator: Jacobian" << std::endl;
                RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromJacobian< MattesMutualInformationMetricType > >(input);
                break;            

        }
    }

    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
            // std::cout << "EXIT_FAILURE = " << EXIT_FAILURE << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
 