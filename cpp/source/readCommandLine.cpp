/**
 * \brief Implementation of the function readCommandLine.
 *
 * readCommandLine parses the sInput parameters given by the command line.
 *
 * \author Michael Ebner
 * \date April 2015
 */
 
#include "readCommandLine.h"

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

namespace po = boost::program_options;

std::vector<std::string> readCommandLine(int argc, char** argv){

    std::vector<std::string> sInput;
    
    std::string sFixed;
    std::string sMoving;
    std::string sFixedMask;
    std::string sMovingMask;
    std::vector<std::string> sPsfCov;
    std::string sPsfCov_all;
    std::string sUseMultiresolution;
    std::string sUseAffine;
    std::string sMetric;
    std::string sInterpolator;
    std::string sScalesEstimator;
    std::string sTransformOut;
    std::string sVerbose;
    std::string sANTSrad;
    std::string sTranslationScale;
    // std::string sOptimizer; //TODO

    const bool bVerbose = false; // verbose for this file

    const std::string sBar = "------------------------------------------------------" 
        "----------------------------\n";

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("f", po::value< std::vector<std::string> >(), 
        "specify filename of fixed image, \n"
        "e.g. \"--f path-to-file/fixed\" without '.nii.gz' extension")
    ("m", po::value< std::vector<std::string> >(), 
        "specify filename of moving image, \n"
        "e.g. \"--m path-to-file/moving\" without '.nii.gz' extension")
    ("fmask", po::value< std::vector<std::string> >(), 
        "specify filename of fixed image mask (optional), \n"
        "e.g. \"--m path-to-file/fixed_mask\" without '.nii.gz' extension")
    ("mmask", po::value< std::vector<std::string> >(), 
        "specify filename of moving image mask (optional), \n"
        "e.g. \"--m path-to-file/moving_mask\" without '.nii.gz' extension")
    ("useMultires", po::value< std::vector<std::string> >(),
        "specify whether multiresolution framework is desired (default: 0), \n"
        "e.g. \"--useMultires 1\"")
    ("useAffine", po::value< std::vector<std::string> >(), 
        "specify whether affine registration is desired (default: rigid registration), \n"
        "e.g. \"--useAffine 1\"")
    ("metric", po::value< std::vector<std::string> >(), 
        "specify which metric shall be chosen (default: MattesMutualInformation), \n"
        "e.g. \"--metric MeanSquares\", MattesMutualInformation or Correlation")
    ("interpolator", po::value< std::vector<std::string> >(), 
        "specify which interpolator shall be chosen (default: Linear), \n"
        "e.g. \"--interpolator NearestNeighbor\", Linear, BSpline, Gaussian or OrientedGaussian (set via 'cov')")
    ("scalesEst", po::value< std::vector<std::string> >(), 
        "specify which scales estimator shall be chosen (default: Jacobian), \n"
        "e.g. \"--scalesEst PhysicalShift\", ScalesShift, PhysicalShift, Jacobian")
    // ("cov", po::value< std::vector<std::string> >(&sPsfCov)->multitoken(), 
    //     "specify the PSF in case oriented Gaussian interpolation is desired (optional), \n"
    //     "e.g. \"--cov cov_11 cov_12 cov_13 cov_21 cov_22 cov_23 cov_31 cov_32 cov_33\"")
    ("cov", po::value< std::vector<std::string> >(), 
        "specify the PSF in case oriented Gaussian interpolation is desired (optional), \n"
        "e.g. \"--cov cov_11 cov_12 cov_13 cov_21 cov_22 cov_23 cov_31 cov_32 cov_33\"")
    ("tout", po::value< std::vector<std::string> >(), 
        "specify the file to write obtained registration transform (optional), \n"
        "e.g. \"--tout /tmp/transform.txt\"")
    ("verbose", po::value< std::vector<std::string> >(), 
        "specify whether full output is desired (default=1), \n"
        "e.g. \"--verbose 0\"")
    ("ANTSrad", po::value< std::vector<std::string> >(), 
        "specify radius used for ANTSNeighborhoodCorrelation (default=20), \n"
        "e.g. \"--ANTSrad 30\"")
    ("translationScale", po::value< std::vector<std::string> >(), 
        "specify scale used for translation (default=1), \n"
        "e.g. \"--translationScale 10\"")
    // ("optimizer", po::value< std::vector<std::string> >(), 
    //     "specify optimizer (default: RegularStepGradientDescent), \n"
    //     "e.g. \"--optimizer RegularStepGradientDescent\"")
    ;

    po::variables_map vm;        
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("verbose")) {
        sVerbose = vm["verbose"].as< std::vector<std::string> >()[0];
    }
    else {
        sVerbose = "0";
    }

    if ( bVerbose ) {
        std::cout << sBar;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        sInput.push_back("help request");
        std::cout << sBar;
        return sInput;
    }


    if (vm.count("f")) {
        if ( bVerbose ) {
            std::cout << "fixed image given (" 
                << vm["f"].as< std::vector<std::string> >()[0] << ".nii.gz).\n";
        }
        sFixed = vm["f"].as< std::vector<std::string> >()[0];
    } 
    else {
        throw ExceptionCommandLine("fixed image not given.");
    }

    if (vm.count("m")) {
        if ( bVerbose ) {
            std::cout << "moving image given (" 
                << vm["m"].as< std::vector<std::string> >()[0] << ".nii.gz).\n";
        }
        sMoving = vm["m"].as< std::vector<std::string> >()[0];

        // //***Check whether output file already exists:
        // bool fileExists = ( access( sfileGameHistory.c_str(), F_OK ) != -1 );
        // if (fileExists){
        //     throw ExceptionCommandLine("Output file already exists.");
        // }

    } 
    else {
        throw ExceptionCommandLine("moving image not given.");
    }

    if (vm.count("fmask")) {
        if ( bVerbose ) {
            std::cout << "fixed image mask given (" 
                << vm["fmask"].as< std::vector<std::string> >()[0] << ".nii.gz).\n";
        }
        sFixedMask = vm["fmask"].as< std::vector<std::string> >()[0];
    }
    else {
        sFixedMask = "";
    }

    if (vm.count("mmask")) {
        if ( bVerbose ) {
            std::cout << "moving image mask given (" 
                << vm["mmask"].as< std::vector<std::string> >()[0] << ".nii.gz).\n";
        }
        sMovingMask = vm["mmask"].as< std::vector<std::string> >()[0];
    }
    else {
        sMovingMask = "";
    }

    if (vm.count("cov")) {
        sPsfCov_all = vm["cov"].as< std::vector<std::string> >()[0];
        split(sPsfCov, sPsfCov_all, boost::is_any_of(" "));
        if ( bVerbose ) {
            std::cout << "covariance for oriented Gaussian = " << std::endl;
            for (int i = 0; i < 3; ++i) {
                printf("\t%s\t%s\t%s\n", sPsfCov[3*i].c_str(), sPsfCov[3*i+1].c_str(), sPsfCov[3*i+2].c_str());
            }
            std::cout <<  vm["cov"].as< std::vector<std::string> >()[0] << std::endl;
            std::cout << "PSF covariance for oriented Gaussian interpolator given" << std::endl;  
            // std::cout << "\t cov (1,1) = " << sPsfCov[0] << "\n";
            // std::cout << "\t cov (1,2) = " << sPsfCov[1] << "\n";
            // std::cout << "\t cov (1,3) = " << sPsfCov[2] << "\n";
            // std::cout << "\t cov (2,1) = " << sPsfCov[3] << "\n";
            // std::cout << "\t cov (2,2) = " << sPsfCov[4] << "\n";
            // std::cout << "\t cov (2,3) = " << sPsfCov[5] << "\n";
            // std::cout << "\t cov (3,1) = " << sPsfCov[6] << "\n";
            // std::cout << "\t cov (3,2) = " << sPsfCov[7] << "\n";
            // std::cout << "\t cov (3,3) = " << sPsfCov[8] << "\n";
        }
    }
    // Initialize by identity if not given
    else{
        sPsfCov.push_back("1");
        sPsfCov.push_back("0");
        sPsfCov.push_back("0");
        sPsfCov.push_back("0");
        sPsfCov.push_back("1");
        sPsfCov.push_back("0");
        sPsfCov.push_back("0");
        sPsfCov.push_back("0");
        sPsfCov.push_back("1");
    }

    if (vm.count("useMultires")) {
        if ( bVerbose ) {
            std::cout << "Multiresolution flag is set to " 
                << vm["useMultires"].as< std::vector<std::string> >()[0] << std::endl;  
        }
        sUseMultiresolution = vm["useMultires"].as< std::vector<std::string> >()[0];
    }
    // No multi-resolution framework by default
    else {
        sUseMultiresolution = "0";
    }

    if (vm.count("useAffine")) {
        if ( bVerbose ) {
            std::cout << "useAffine flag is set to " 
                << vm["useAffine"].as< std::vector<std::string> >()[0] << std::endl;  
        }
        sUseAffine = vm["useAffine"].as< std::vector<std::string> >()[0];
    }
    // Rigid registration by default
    else {
        sUseAffine = "0";
    }

    if (vm.count("metric")) {
        if ( bVerbose ) {
            std::cout << "chosen metric is " 
                << vm["metric"].as< std::vector<std::string> >()[0] << std::endl;  
        }
        sMetric = vm["metric"].as< std::vector<std::string> >()[0];
    }
    // MattesMutualInformation by default
    else {
        sMetric = "MattesMutualInformation";
    }

    if (vm.count("interpolator")) {
        if ( bVerbose ) {
            std::cout << "chosen interpolator is " 
                << vm["interpolator"].as< std::vector<std::string> >()[0] << std::endl;  
        }
        sInterpolator = vm["interpolator"].as< std::vector<std::string> >()[0];
    }
    // BSpline interpolator by default
    else {
        sInterpolator = "BSpline";
    }

    if (vm.count("scalesEst")) {
        if ( bVerbose ) {
            std::cout << "chosen scales estimator is " 
                << vm["scalesEst"].as< std::vector<std::string> >()[0] << std::endl;  
        }
        sScalesEstimator = vm["scalesEst"].as< std::vector<std::string> >()[0];
    }
    // BSpline interpolator by default
    else {
        sScalesEstimator = "Jacobian";
    }

    if (vm.count("tout")) {
        if ( bVerbose ) {
            std::cout << "chosen file to write obtained registration transform is " 
                << vm["tout"].as< std::vector<std::string> >()[0] << std::endl;  
        }
        sTransformOut = vm["tout"].as< std::vector<std::string> >()[0];
    }
    // No output by default
    else {
        sTransformOut = "";
    }

    if (vm.count("ANTSrad")) {
        if ( bVerbose ) {
            std::cout << "chosen radius for ANTSNeighborhoodCorrelation is " 
                << vm["ANTSrad"].as< std::vector<std::string> >()[0] << std::endl;  
        }
        sANTSrad = vm["ANTSrad"].as< std::vector<std::string> >()[0];
    }
    // radius by default
    else {
        sANTSrad = "20";
    }

    if (vm.count("translationScale")) {
        if ( bVerbose ) {
            std::cout << "chosen scale for translation is " 
                << vm["translationScale"].as< std::vector<std::string> >()[0] << std::endl;  
        }
        sTranslationScale = vm["translationScale"].as< std::vector<std::string> >()[0];
    }
    // radius by default
    else {
        sTranslationScale = "1";
    }


    if ( bVerbose ) {
        std::cout << sBar;
    }

    // // Exactly three sInput parameters (i.e. 2*2+1 arguments) must be given
    // if (argc != 5){
    //     // std::string msg = "Number of sInput arguments = " 
    //     // + std::to_string((unsigned int) (argc-1)/2) + ". "
    //     throw ExceptionCommandLine("sInput arguments are not correct.");
    // }

    sInput.push_back(sFixed);
    sInput.push_back(sMoving);
    sInput.push_back(sFixedMask);
    sInput.push_back(sMovingMask);

    for (int i = 0; i < 9; ++i) {
        sInput.push_back(sPsfCov[i]);
        // std::cout << sPsfCov[i] << std::endl;
    }

    sInput.push_back(sUseMultiresolution);
    sInput.push_back(sUseAffine);
    sInput.push_back(sMetric);
    sInput.push_back(sInterpolator);
    sInput.push_back(sTransformOut);
    sInput.push_back(sScalesEstimator);
    sInput.push_back(sVerbose);
    sInput.push_back(sANTSrad);
    sInput.push_back(sTranslationScale);


    return sInput;
}