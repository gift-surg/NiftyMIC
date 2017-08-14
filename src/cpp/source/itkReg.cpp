/*! \brief
 *
 *  
 *
 *  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
 *  \date May 2016
 */
#include <boost/type_traits.hpp>

#include <iostream>
#include <sstream>
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
#include <itkMultiStartOptimizerv4.h>
#include <itkConjugateGradientLineSearchOptimizerv4.h>

#include <itkResampleImageFilter.h>
// #include <itkRescaleIntensityImageFilter.h>

#include <itkAffineTransform.h>
#include <itkEuler3DTransform.h>
#include <itkImageMaskSpatialObject.h>

#include <itkRegistrationParameterScalesEstimator.h>
#include <itkRegistrationParameterScalesFromJacobian.h>
#include <itkRegistrationParameterScalesFromIndexShift.h>
#include <itkRegistrationParameterScalesFromPhysicalShift.h>

#include <itkCommand.h>

// My includes
#include "MyITKImageHelper.h"
#include "itkOrientedGaussianInterpolateImageFunction.h"
#include "readCommandLine.h"
#include "MyException.h"
// #include "itkScaledTranslationEuler3DTransform.h"

// Global variables
const unsigned int Dimension = 3;

// Typedefs 
typedef itk::ResampleImageFilter< ImageType3D, ImageType3D > ResampleFilterType;
typedef itk::ResampleImageFilter< MaskImageType3D, MaskImageType3D > MaskResampleFilterType;

typedef itk::ImageMaskSpatialObject< Dimension > MaskType;

// Transform Types
typedef itk::AffineTransform< PixelType, Dimension > AffineTransformType;
typedef itk::Euler3DTransform< PixelType > EulerTransformType;
// typedef itk::ScaledTranslationEuler3DTransform< PixelType > ScaledTranslationEulerTransformType;
// typedef ScaledTranslationEulerTransformType EulerTransformType;

// Optimizer Types
typedef itk::RegularStepGradientDescentOptimizerv4< PixelType > RegularStepGradientDescentOptimizerType;
typedef itk::LBFGSBOptimizerv4 LBFGSBOptimizerOptimizerType;
typedef itk::MultiStartOptimizerv4 MultiStartOptimizerType;
typedef itk::ConjugateGradientLineSearchOptimizerv4Template< PixelType > ConjugateGradientLineSearchOptimizerType;

// typedef LBFGSBOptimizerOptimizerType OptimizerType;
// typedef MultiStartOptimizerType OptimizerType;
// typedef RegularStepGradientDescentOptimizerType OptimizerType;
typedef ConjugateGradientLineSearchOptimizerType OptimizerType;

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
    shrinkFactorsPerLevel.SetSize( numberOfLevels );
    shrinkFactorsPerLevel[0] = 4;
    shrinkFactorsPerLevel[1] = 2;
    shrinkFactorsPerLevel[2] = 1;

    smoothingSigmasPerLevel.SetSize( numberOfLevels );
    smoothingSigmasPerLevel[0] = 2;
    smoothingSigmasPerLevel[1] = 1;
    smoothingSigmasPerLevel[2] = 0;


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
    const bool bVerbose = std::stoi(sVerbose);
    const double dANTSrad = std::stod(input[20]);

    // Helper to turn on/off verbose output
    std::stringstream ss;
    const bool bDebug = bVerbose;
    // const bool bDebug = false;

    // Read images
    const ImageType3D::Pointer moving = MyITKImageHelper::readImage<ImageType3D>(sMoving);
    const ImageType3D::Pointer fixed = MyITKImageHelper::readImage<ImageType3D>(sFixed);
    // ss.str(""); ss << "Fixed image  = " << sFixed;
    // MyITKImageHelper::printInfo(ss.str(), bDebug);
    // ss.str(""); ss << "Moving image = " << sMoving;
    // MyITKImageHelper::printInfo(ss.str(), bDebug);

    // MyITKImageHelper::showImage(moving, "moving");
    // MyITKImageHelper::showImage(fixed, fixedMask, "fixed");

    // metric->SetUseFixedImageGradientFilter(true);
    // metric->SetUseMovingImageGradientFilter(true);

    // Set registration
    registration->SetFixedImage(fixed);
    registration->SetMovingImage(moving);
    registration->SetMetric( metric );
    registration->SetOptimizer( optimizer );

    registration->SetMetricSamplingPercentage(1.0);
    // registration->MetricSamplingReinitializeSeed(1);
    registration->MetricSamplingReinitializeSeed();

    // Initialize the transform
    typename TransformType::Pointer initialTransform = TransformType::New();
    initialTransform->SetIdentity();

    typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();
    initializer->SetTransform(initialTransform);
    initializer->SetFixedImage( fixed );
    initializer->SetMovingImage( moving );
    if (0){
        initializer->GeometryOn();
        // initializer->MomentsOn();
        initializer->InitializeTransform();
    }
    // initialTransform->Print(std::cout);
    // initialTransform->SetTranslation((0,0,0));
    // initialTransform->Print(std::cout);
    // initialTransform->SetFixedParameters(foo->GetFixedParameters());    
    registration->SetInitialTransform( initialTransform );
    registration->InPlaceOn();
    // registration->GetInitialTransform()->Print(std::cout);

    // // Set scale for translation if itkScaledTranslationEuler3DTransform
    // ScaledTranslationEulerTransformType::Pointer scaledTranslationTransform = dynamic_cast< ScaledTranslationEulerTransformType* >(registration->GetModifiableTransform());
    // if ( scaledTranslationTransform.IsNotNull() ) {
    //     scaledTranslationTransform->SetTranslationScale( dTranslationScale );
    //     ss.str(""); ss << "TranslationScale: " << scaledTranslationTransform->GetTranslationScale();
    //     MyITKImageHelper::printInfo(ss.str(), bDebug);
    // }

    // Read masks
    if(!sFixedMask.empty()){
        // ss.str(""); ss << "Fixed mask image = " << sFixedMask;
        ss.str(""); ss << "Fixed mask used";
        MyITKImageHelper::printInfo(ss.str(), bDebug);
        bUseFixedMask = true;
        fixedMask = MyITKImageHelper::readImage<MaskImageType3D>(sFixedMask);
        spatialObjectFixedMask->SetImage( fixedMask );
        metric->SetFixedImageMask( spatialObjectFixedMask );
    }
    if(!sMovingMask.empty()){
        // ss.str(""); ss << "Moving mask image = " << sMovingMask;
        ss.str(""); ss << "Moving mask used";
        MyITKImageHelper::printInfo(ss.str(), bDebug);
        bUseMovingMask = true;
        movingMask = MyITKImageHelper::readImage<MaskImageType3D>(sMovingMask);
        spatialObjectMovingMask->SetImage( movingMask );
        metric->SetMovingImageMask( spatialObjectMovingMask );
    }

    // Info output transform
    // if(!sTransformOut.empty()){
    //     ss.str(""); ss << "Output transform = " << sTransformOut;
    //     MyITKImageHelper::printInfo(ss.str(), bDebug);
    // }
    
    // Multi-resolution framework
    if (std::stoi(sUseMultiresolution)) {
        bUseMultiresolution = true;
        ss.str(""); ss << "Multiresolution framework used";
        MyITKImageHelper::printInfo(ss.str(), bDebug);

        registration->SetNumberOfLevels ( numberOfLevels );
        registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
        registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
        registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits( true );
    }
    // Multi-resolution framework is used by default! Update to not use it
    else{
        shrinkFactorsPerLevel.SetSize( 1 );
        shrinkFactorsPerLevel[0] = 1;
        smoothingSigmasPerLevel.SetSize( 1 );
        smoothingSigmasPerLevel[0] = 0;

        registration->SetNumberOfLevels ( 1 );
        registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
        registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
        registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits( true );
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
        ss.str(""); ss << "Radius for ANTSNeighborhoodCorrelation = " << dANTSrad;
        MyITKImageHelper::printInfo(ss.str(), bDebug);
        // getting derivative and metric value
        // ANTSmetric->GetValueAndDerivative(valueReturn, derivativeReturn);
    }

    // Set oriented Gaussian interpolator (if given)
    OrientedGaussianInterpolatorType::Pointer orientedGaussianInterpolator = dynamic_cast< OrientedGaussianInterpolatorType* >(interpolator.GetPointer());
    if ( orientedGaussianInterpolator.IsNotNull() ) {
        orientedGaussianInterpolator->SetCovariance( covariance );
        orientedGaussianInterpolator->SetAlpha( 3 );
        // ss.str(""); ss << "OrientedGaussianInterpolator updated ";
        MyITKImageHelper::printInfo(ss.str(), bDebug);
        ss.str(""); ss << "Covariance for oriented Gaussian = ";
        MyITKImageHelper::printInfo(ss.str(), bDebug);
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

    // Set metric
    metric->SetMovingInterpolator(  interpolator  );
    // metric->SetFixedInterpolator(  interpolator->Clone()  );
    
    // std::cout<<"metric->GetUseMovingImageGradientFilter() = " << (metric->GetUseMovingImageGradientFilter()?"True":"False") <<std::endl;
    // std::cout<<"metric->GetMovingImageGradientFilter() = ";
    // metric->GetMovingImageGradientFilter()->Print(std::cout);
    // std::cout<<"metric->GetMovingImageGradientCalculator() = ";
    // metric->GetMovingImageGradientCalculator()->Print(std::cout);
    //std::cout<<"metric->GetUseMovingImageGradientFilter() = " << (metric->GetUseMovingImageGradientFilter()?"True":"False") << std::endl;

    // Scales estimator
    // scalesEstimator->UnRegister();
    scalesEstimator->SetMetric( metric );
    scalesEstimator->SetTransformForward( true );
    typename itk::RegistrationParameterScalesFromPhysicalShift<MetricType>::Pointer myScalesEstimator = dynamic_cast<
        itk::RegistrationParameterScalesFromPhysicalShift<MetricType>* >(scalesEstimator.GetPointer());
    if (myScalesEstimator.IsNotNull()){
        myScalesEstimator->SetCentralRegionRadius(5);
        myScalesEstimator->SetSmallParameterVariation(0.01);
    }
    scalesEstimator->Register();

    // For Regular Step Gradient Descent Optimizer
    RegularStepGradientDescentOptimizerType::Pointer optimizerRegularStep = dynamic_cast<RegularStepGradientDescentOptimizerType* > (optimizer.GetPointer());
    if ( optimizerRegularStep.IsNotNull() ){
        optimizerRegularStep->SetScalesEstimator( scalesEstimator );
        optimizerRegularStep->SetLearningRate(1);
        optimizerRegularStep->SetMinimumStepLength( 1e-6 );
        optimizerRegularStep->SetNumberOfIterations( 500 );
        optimizerRegularStep->SetRelaxationFactor( 0.5 );
        optimizerRegularStep->SetGradientMagnitudeTolerance( 1e-6 );
        optimizerRegularStep->SetDoEstimateLearningRateOnce( false );
        optimizerRegularStep->SetMaximumStepSizeInPhysicalUnits( 0.0 );
        ss.str(""); ss << "Optimizer: RegularStepGradientDescentOptimizerv4";
        MyITKImageHelper::printInfo(ss.str(), bDebug);

    }

    // For ConjugateGradientLineSearch Optimizer
    ConjugateGradientLineSearchOptimizerType::Pointer optimizerCGLS = dynamic_cast<ConjugateGradientLineSearchOptimizerType* > (optimizer.GetPointer());
    if ( optimizerCGLS.IsNotNull() ){
        // Based on SimpleITK default settings
        optimizerCGLS->SetScalesEstimator( scalesEstimator );
        optimizerCGLS->SetLearningRate( 1 );
        optimizerCGLS->SetNumberOfIterations( 100 );
        optimizerCGLS->SetConvergenceWindowSize( 10 );
        optimizerCGLS->SetMinimumConvergenceValue( 1e-6 );
        optimizerCGLS->SetLowerLimit( 0 );
        optimizerCGLS->SetUpperLimit( 5 );
        optimizerCGLS->SetEpsilon( 0.01 );
        optimizerCGLS->SetMaximumLineSearchIterations( 20 );
        optimizerCGLS->SetMaximumStepSizeInPhysicalUnits( 0.0 );
        optimizerCGLS->SetDoEstimateLearningRateOnce( true );
        // optimizerCGLS->SetDoEstimateLearningRateAtEachIteration( true );

        // optimizerCGLS->SetDoEstimateLearningRateAtEachIteration( false );
        ss.str(""); ss << "Optimizer: ConjugateGradientLineSearchOptimizerv4";
        MyITKImageHelper::printInfo(ss.str(), bDebug);
    }

    // For LBFGS Optimizer
    LBFGSBOptimizerOptimizerType::Pointer optimizerLBFGS = dynamic_cast<LBFGSBOptimizerOptimizerType* > (optimizer.GetPointer());
    if ( optimizerLBFGS.IsNotNull() ){
        const unsigned int numParameters = initialTransform->GetNumberOfParameters();

        LBFGSBOptimizerOptimizerType::BoundSelectionType boundSelect( numParameters );
        LBFGSBOptimizerOptimizerType::BoundValueType upperBound( numParameters );
        LBFGSBOptimizerOptimizerType::BoundValueType lowerBound( numParameters );
        boundSelect.Fill( LBFGSBOptimizerOptimizerType::BOTHBOUNDED );
        upperBound.Fill( 0.0 );
        lowerBound.Fill( 0.0 );

        const double angle_deg_max = 5.0;
        const double translation_max = 10.0;
        for (int i = 0; i < 3; ++i) {
            lowerBound[i] = -angle_deg_max*vnl_math::pi/180;
            upperBound[i] =  angle_deg_max*vnl_math::pi/180;
            
            lowerBound[i+3] = -translation_max;
            upperBound[i+3] =  translation_max;
        }

        optimizerLBFGS->SetBoundSelection( boundSelect );
        optimizerLBFGS->SetUpperBound( upperBound );
        optimizerLBFGS->SetLowerBound( lowerBound );

        optimizerLBFGS->SetCostFunctionConvergenceFactor( 1.e7 );
        optimizerLBFGS->SetGradientConvergenceTolerance( 1e-35 );
        optimizerLBFGS->SetNumberOfIterations( 200 );
        optimizerLBFGS->SetMaximumNumberOfFunctionEvaluations( 200 );
        optimizerLBFGS->SetMaximumNumberOfCorrections( 7 );
        ss.str(""); ss << "Optimizer: LBFGSBOptimizerv4";
        MyITKImageHelper::printInfo(ss.str(), bDebug);
    }


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

    // Debug
    // registration->Print(std::cout);
    // registration->GetOptimizer()->Print(std::cout);
    // registration->GetMetric()->Print(std::cout);

    //***Execute registration
    try {
        registration->Update();

        if (bVerbose) {
            ss.str(""); ss << "Summary RegistrationCppITK: ";
            MyITKImageHelper::printInfo(ss.str(), bDebug);

            ss.str(""); ss << "\tOptimizer\'s stopping condition: "
            << registration->GetOptimizer()->GetStopConditionDescription();
            MyITKImageHelper::printInfo(ss.str(), bDebug);

            ss.str(""); ss << "\tFinal metric value: " << optimizer->GetValue();
            MyITKImageHelper::printInfo(ss.str(), bDebug);
        }
    }
    catch( itk::ExceptionObject & err ) {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      // return EXIT_FAILURE;
      throw MyException("ExeceptionObject caught during registration");
    }


    //***Process registration results
    typename TransformType::ConstPointer transform = registration->GetTransform();
    
    if ( bVerbose ) {
        // transform->Print(std::cout);
        MyITKImageHelper::printTransform(transform);
    }

    //***Write result to file
    if ( !sTransformOut.empty() ) {
        MyITKImageHelper::writeTransform(transform, sTransformOut, 0);
    }

    //***Resample warped moving image
    if (0){
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
        resampler->SetInterpolator( LinearInterpolatorType::New() );
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

        // Remove extension from filename
        size_t lastindex = sTransformOut.find_last_of("."); 
        const std::string sTransformOutWithoutExtension = sTransformOut.substr(0, lastindex);
        MyITKImageHelper::writeImage(movingWarped, sTransformOutWithoutExtension + "warpedMoving.nii.gz", bDebug);
        // MyITKImageHelper::writeImage(movingMaskWarped, sTransformOut + "warpedMoving_mask.nii.gz");

        // MyITKImageHelper::showImage(fixed, movingWarped, "fixed_moving");
        // MyITKImageHelper::showImage(fixed, movingWarped, movingWarped, "fixed_moving");
        // MyITKImageHelper::showImage(movingWarped, movingMaskWarped, "fixed_mask");
    }
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

        // Information on parametrization on/off
        const bool bDebug = std::stoi(input[19]);
        // const bool bDebug = false;
        std::stringstream ss;

        ss.str(""); ss << "Registration: CppITK";
        MyITKImageHelper::printInfo(ss.str(), bDebug);

        // TODO: At the moment only rigid model is available
        switch ( std::stoi(sUseAffine) ){
            
            // Rigid registration
            case 0:
                ss.str(""); ss << "Transform Model: Rigid";
                MyITKImageHelper::printInfo(ss.str(), bDebug);

                // Nearest Neighbor interpolator
                if ( sInterpolator == ("NearestNeighbor") ) {
                    ss.str(""); ss << "Interpolator: " << sInterpolator;
                    MyITKImageHelper::printInfo(ss.str(), bDebug);

                    // Mean Squares metric
                    if ( sMetric == ("MeanSquares") ) { 
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MeanSquaresMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromIndexShift< MeanSquaresMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromJacobian< MeanSquaresMetricType > >(input);
                        }

                    }

                    // Normalized Cross Correlation Metric
                    else if ( sMetric == ("Correlation") ){
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< CorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< CorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< CorrelationMetricType > >(input);
                        }

                    }

                    // ANTS Neighborhood Correlation Metric
                    else if ( sMetric == ("ANTSNeighborhoodCorrelation") ){
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< ANTSNeighborhoodCorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                    }

                    // Mattes Mutual Information Metric
                    else {
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MattesMutualInformationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromIndexShift< MattesMutualInformationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, NearestNeighborInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromJacobian< MattesMutualInformationMetricType > >(input);
                        }
                    }
                }

                // Linear interpolator
                else if ( sInterpolator == ("Linear") ) {
                    ss.str(""); ss << "Interpolator: " << sInterpolator;
                    MyITKImageHelper::printInfo(ss.str(), bDebug);


                    // Mean Squares metric
                    if ( sMetric == ("MeanSquares") ) { 
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);
                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MeanSquaresMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromIndexShift< MeanSquaresMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromJacobian< MeanSquaresMetricType > >(input);
                        }

                    }

                    // Normalized Cross Correlation Metric
                    else if ( sMetric == ("Correlation") ){
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< CorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< CorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< CorrelationMetricType > >(input);
                        }

                    }

                    // ANTS Neighborhood Correlation Metric
                    else if ( sMetric == ("ANTSNeighborhoodCorrelation") ){
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< ANTSNeighborhoodCorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                    }

                    // Mattes Mutual Information Metric
                    else {
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MattesMutualInformationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromIndexShift< MattesMutualInformationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, LinearInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromJacobian< MattesMutualInformationMetricType > >(input);
                        }
                    }
                }

                // Oriented Gaussian interpolator
                else if ( sInterpolator == ("OrientedGaussian") ) {
                    ss.str(""); ss << "Interpolator: " << sInterpolator;
                    MyITKImageHelper::printInfo(ss.str(), bDebug);

                    // Mean Squares metric
                    if ( sMetric == ("MeanSquares") ) { 
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MeanSquaresMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromIndexShift< MeanSquaresMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromJacobian< MeanSquaresMetricType > >(input);
                        }

                    }

                    // Normalized Cross Correlation Metric
                    else if ( sMetric == ("Correlation") ){
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< CorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< CorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< CorrelationMetricType > >(input);
                        }

                    }

                    // ANTS Neighborhood Correlation Metric
                    else if ( sMetric == ("ANTSNeighborhoodCorrelation") ){
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< ANTSNeighborhoodCorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                    }

                    // Mattes Mutual Information Metric
                    else {
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MattesMutualInformationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromIndexShift< MattesMutualInformationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, OrientedGaussianInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromJacobian< MattesMutualInformationMetricType > >(input);
                        }
                    }
                }

                // BSpline interpolator
                else {
                    ss.str(""); ss << "Interpolator: BSpline";
                    MyITKImageHelper::printInfo(ss.str(), bDebug);

                    // Mean Squares metric
                    if ( sMetric == ("MeanSquares") ) { 
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MeanSquaresMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromIndexShift< MeanSquaresMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MeanSquaresMetricType, itk::RegistrationParameterScalesFromJacobian< MeanSquaresMetricType > >(input);
                        }

                    }

                    // Normalized Cross Correlation Metric
                    else if ( sMetric == ("Correlation") ){
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< CorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< CorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, CorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< CorrelationMetricType > >(input);
                        }

                    }

                    // ANTS Neighborhood Correlation Metric
                    else if ( sMetric == ("ANTSNeighborhoodCorrelation") ){
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< ANTSNeighborhoodCorrelationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromIndexShift< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, ANTSNeighborhoodCorrelationMetricType, itk::RegistrationParameterScalesFromJacobian< ANTSNeighborhoodCorrelationMetricType > >(input);
                        }

                    }

                    // Mattes Mutual Information Metric
                    else {
                        ss.str(""); ss << "Metric: " << sMetric;
                        MyITKImageHelper::printInfo(ss.str(), bDebug);

                        // Physical Shift step estimator
                        if ( sScalesEstimator == ("PhysicalShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);
                            
                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromPhysicalShift< MattesMutualInformationMetricType > >(input);

                        }
                        // Index Shift step estimator
                        else if ( sScalesEstimator == ("IndexShift") ) {
                            ss.str(""); ss << "Optimizer Scales Estimator: " << sScalesEstimator;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromIndexShift< MattesMutualInformationMetricType > >(input);
                        }

                        // Jacobian step estimator
                        else {
                            ss.str(""); ss << "Optimizer Scales Estimator: Jacobian" ;
                            MyITKImageHelper::printInfo(ss.str(), bDebug);

                            RegistrationFunction<EulerTransformType, BSplineInterpolatorType, MattesMutualInformationMetricType, itk::RegistrationParameterScalesFromJacobian< MattesMutualInformationMetricType > >(input);
                        }
                    }
                }

                break;


            // TODO: Same as above but replace EulerTransformType by AffineTransformType.
            // However, write test cases first!!!
            case 1:
                // ss.str(""); ss << "Affine registration used";
                // MyITKImageHelper::printInfo(ss.str(), bDebug);
                std::cerr << "Error: Affine registration not implemented yet." << "\n";
                return EXIT_FAILURE;
                break;

            default:

                // // Scales Estimator Types
                // typedef  PhysicalShiftScalesEstimatorType;
                // typedef itk::RegistrationParameterScalesFromIndexShift< MetricType > IndexShiftScalesEstimatorType;
                // typedef itk::RegistrationParameterScalesFromJacobian< MetricType > JacobianScalesEstimatorType;

                ss.str(""); ss << "Transform Model: Rigid";
                MyITKImageHelper::printInfo(ss.str(), bDebug);
                ss.str(""); ss << "Interpolator: BSpline";
                MyITKImageHelper::printInfo(ss.str(), bDebug);
                ss.str(""); ss << "Metric: Mattes Mutual Information";
                MyITKImageHelper::printInfo(ss.str(), bDebug);
                ss.str(""); ss << "Optimizer Scales Estimator: Jacobian";
                MyITKImageHelper::printInfo(ss.str(), bDebug);
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
 