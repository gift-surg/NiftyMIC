/**
 * \brief Implementation of the function readCommandLine.
 *
 * readCommandLine parses the sInput parameters given by the command line.
 *
 * \author Michael Ebner
 * \date April 2015
 */
 
#include "readCommandLine.h"

namespace po = boost::program_options;

std::vector<std::string> readCommandLine(int argc, char** argv){

    std::vector<std::string> sInput;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("f", po::value< std::vector<std::string> >(), 
        "specify filename of fixed image, \n"
        "e.g. \"--f path-to-file/fixed\" without '.nii.gz' extension")
    ("m", po::value< std::vector<std::string> >(), 
        "specify filename of moving image, \n"
        "e.g. \"--m path-to-file/moving\" without '.nii.gz' extension")
    ;

    po::variables_map vm;        
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    const std::string sBar = "------------------------------------------------------" 
        "----------------------------\n";
    
    std::string sMoving;
    std::string sFixed;
    std::cout << sBar;

    if (vm.count("help")) {
        std::cout << desc << "\n";
        sInput.push_back("help request");
        std::cout << sBar;
        return sInput;
    }

    if (vm.count("f")) {
        std::cout << "fixed image given (" 
            << vm["f"].as< std::vector<std::string> >()[0] << ".nii.gz).\n";
        sFixed = vm["f"].as< std::vector<std::string> >()[0];
    } 
    else {
        throw ExceptionCommandLine("fixed image not given.");
    }

    if (vm.count("m")) {
        std::cout << "moving image given (" 
            << vm["m"].as< std::vector<std::string> >()[0] << ".nii.gz).\n";

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

    std::cout << sBar;

    // Exactly three sInput parameters (i.e. 2*2+1 arguments) must be given
    if (argc != 5){
        // std::string msg = "Number of sInput arguments = " 
        // + std::to_string((unsigned int) (argc-1)/2) + ". "
        throw ExceptionCommandLine("sInput arguments are not correct.");
    }

    sInput.push_back(sFixed);
    sInput.push_back(sMoving);

    return sInput;
}