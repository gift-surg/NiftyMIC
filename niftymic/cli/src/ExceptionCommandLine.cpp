/**
 * \brief Implementation of the ExceptionCommandLine class.
 *
 * Exception CommandLine deals with the exceptions related to the parsing of
 * invalid input parameters via the command line.
 *
 * \author Michael Ebner
 * \date April 2015
 */
 
#include "ExceptionCommandLine.h"

ExceptionCommandLine::ExceptionCommandLine(){
    sErrorMessage = "Please add '--help' for further help.";
};

ExceptionCommandLine::ExceptionCommandLine(std::string str){

    sErrorMessage = str;
    sErrorMessage += "\n\tPlease add '--help' for further help.";
}; 

ExceptionCommandLine::~ExceptionCommandLine() throw() {

};

const char* ExceptionCommandLine::what() const throw() { 
    return this->sErrorMessage.c_str(); 
}
